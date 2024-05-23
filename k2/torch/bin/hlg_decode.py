import argparse
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import k2
import kaldifeat
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from k2 import (
    get_lattice,
    one_best_decoding,
    get_aux_labels,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model",
        type=str,
        required=True,
        help="Path to the jit script model.",
    )

    parser.add_argument(
        "--words-file",
        type=str,
        help="""Path to words.txt.
        Used only when method is not ctc-decoding.
        """,
    )

    parser.add_argument(
        "--HLG",
        type=str,
        help="""Path to HLG.pt.
        Used only when method is not ctc-decoding.
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.
        Used only when method is ctc-decoding.
        """,
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method.
        Possible values are:
        (0) ctc-decoding - Use CTC decoding. It uses a sentence
            piece model, i.e., lang_dir/bpe.model, to convert
            word pieces to words. It needs neither a lexicon
            nor an n-gram LM.
        (1) 1best - Use the best path as decoding output. Only
            the transformer encoder output is used for decoding.
            We call it HLG decoding.
        """,
    )

    parser.add_argument(
        "--wav-scp",
        type=str,
        help="""The audio lists to transcribe in wav.scp format""",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="""
        The file to write out results to, only used when giving --wav-scp
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="The number of wavs in a batch.",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="*",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans



def _intersect_device(
        a_fsas: k2.Fsa,
        b_fsas: k2.Fsa,
        b_to_a_map: torch.Tensor,
        sorted_match_a: bool,
        batch_size: int = 50,
) -> k2.Fsa:
    """This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.

    The arguments and return value of this function are the same as
    :func:`k2.intersect_device`.
    """
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = (num_fsas + batch_size - 1) // batch_size
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index_fsa(b_fsas, indexes)
        b_to_a = k2.index_select(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


class Nbest(object):
    """
    An Nbest object contains two fields:

        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
                 Its axes are [path][state][arc]
        (2) shape. Its type is :class:`k2.RaggedShape`.
                   Its axes are [utt][path]

    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.

    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        """
        Args:
          fsa:
            An FsaVec with axes [path][state][arc]. It is expected to contain
            a list of **linear** FSAs.
          shape:
            A ragged shape with two axes [utt][path].
        """
        assert len(fsa.shape) == 3, f"fsa.shape: {fsa.shape}"
        assert shape.num_axes == 2, f"num_axes: {shape.num_axes}"

        if fsa.shape[0] != shape.tot_size(1):
            raise ValueError(
                f"{fsa.shape[0]} vs {shape.tot_size(1)}\n"
                "Number of FSAs in `fsa` does not match the given shape"
            )

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = "Nbest("
        s += f"Number of utterances:{self.shape.dim0}, "
        s += f"Number of Paths:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
            lattice: k2.Fsa,
            num_paths: int,
            use_double_scores: bool = True,
            nbest_scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.

        Each sampled path is a linear FSA.

        We assume `lattice.labels` contains token IDs and `lattice.aux_labels`
        contains word IDs.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
          use_double_scores:
            True to use double precision in :func:`k2.random_paths`.
            False to use single precision.
          scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
        Returns:
          Return an Nbest instance.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= nbest_scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos]
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        # It axes is [utt][path][word_id]
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)
        word_seq = word_seq.remove_values_leq(0)

        # Each utterance has `num_paths` paths but some of them transduces
        # to the same word sequence, so we need to remove repeated word
        # sequences within an utterance. After removing repeats, each utterance
        # contains different number of paths
        #
        # `new2old` is a 1-D torch.Tensor mapping from the output path index
        # to the input path index.
        _, _, new2old = word_seq.unique(
            need_num_repeats=False, need_new2old_indexes=True
        )

        # kept_path is a ragged tensor with dtype torch.int32.
        # It has axes [utt][path][arc_pos]
        kept_path, _ = path.index(new2old, axis=1, need_value_indexes=False)

        # utt_to_path_shape has axes [utt][path]
        utt_to_path_shape = kept_path.shape.get_layer(0)

        # Remove the utterance axis.
        # Now kept_path has only two axes [path][arc_pos]
        kept_path = kept_path.remove_axis(0)

        # labels is a ragged tensor with 2 axes [path][token_id]
        # Note that it contains -1s.
        labels = k2.ragged.index(lattice.labels.contiguous(), kept_path)

        # Remove -1 from labels as we will use it to construct a linear FSA
        labels = labels.remove_values_eq(-1)

        if isinstance(lattice.aux_labels, k2.RaggedTensor):
            # lattice.aux_labels is a ragged tensor with dtype torch.int32.
            # It has 2 axes [arc][word], so aux_labels is also a ragged tensor
            # with 2 axes [arc][word]
            aux_labels, _ = lattice.aux_labels.index(
                indexes=kept_path.values, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.values)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        # Caution: fsa.scores are all 0s.
        # `fsa` has only one extra attribute: aux_labels.
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice, get 1-best
        path from the resulting FsaVec, and return a new Nbest object.

        The purpose of this function is to attach scores to an Nbest.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume its `labels` are word IDs.
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`. Also, its `fsa` has non-zero scores and inherits attributes
          for `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        word_fsa.scores.zero_()
        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels
            word_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(word_fsa)
        else:
            word_fsa_with_epsilon_loops = k2.linear_fst_with_self_loops(word_fsa)

        path_to_utt_map = self.shape.row_ids(1)

        if hasattr(lattice, "aux_labels"):
            # lattice has token IDs as labels and word IDs as aux_labels.
            # inv_lattice has word IDs as labels and token IDs as aux_labels
            inv_lattice = k2.invert(lattice)
            inv_lattice = k2.arc_sort(inv_lattice)
        else:
            inv_lattice = k2.arc_sort(lattice)

        if inv_lattice.shape[0] == 1:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=torch.zeros_like(path_to_utt_map),
                sorted_match_a=True,
            )
        else:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=path_to_utt_map,
                sorted_match_a=True,
            )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        path_lattice = k2.top_sort(k2.connect(path_lattice))

        one_best = k2.shortest_path(path_lattice, use_double_scores=use_double_scores)

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]
        am_scores = self.fsa.scores - self.fsa.lm_scores
        ragged_am_scores = k2.RaggedTensor(scores_shape, am_scores.contiguous())
        tot_scores = ragged_am_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def compute_lm_scores(self) -> k2.RaggedTensor:
        """Compute LM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_lm_scores = k2.RaggedTensor(
            scores_shape, self.fsa.lm_scores.contiguous()
        )

        tot_scores = ragged_lm_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semiring
          and tropical semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_scores = k2.RaggedTensor(scores_shape, self.fsa.scores.contiguous())

        tot_scores = ragged_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)


def nbest_listing(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        nbest_scale: float = 1.0,
) -> Tuple[k2.RaggedTensor, k2.RaggedTensor, List[List[int]]]:
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # nbest.fsa.scores contains 0s

    nbest = nbest.intersect(lattice)
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()
    lm_scores = nbest.compute_lm_scores()

    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.aux_labels)
    tokens = tokens.remove_values_leq(0)
    tokens = tokens.remove_values_leq(-1)
    token_ids = tokens.tolist()

    return am_scores, lm_scores, token_ids


def decode_one_batch(
    params: object,
    batch: List[Tuple[str, str]],
    model: torch.nn.Module,
    feature_extractor: kaldifeat.Fbank,
    decoding_graph: k2.Fsa,
    token_sym_table: Optional[k2.SymbolTable] = None,
    word_sym_table: Optional[k2.SymbolTable] = None,
) -> Dict[str, str]:
    device = params.device
    filenames = [x[1] for x in batch]
    waves = read_sound_files(
        filenames=filenames, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    features = feature_extractor(waves)

    feature_len = []
    for f in features:
        feature_len.append(f.shape[0])

    features = pad_sequence(
        features, batch_first=True, padding_value=math.log(1e-10)
    )

    # Note: We don't use key padding mask for attention during decoding
    nnet_output, _, _ = model(features)

    log_prob = torch.nn.functional.log_softmax(nnet_output, dim=-1)
    log_prob_len = torch.tensor(feature_len) // params.subsampling_factor
    log_prob_len = log_prob_len.to(device)

    lattice = get_lattice(
        log_prob=log_prob,
        log_prob_len=log_prob_len,
        decoding_graph=decoding_graph,
        subsampling_factor=params.subsampling_factor,
    )
    am_scores, lm_scores, token_ids = nbest_listing(lattice, num_paths=20, nbest_scale=1.0)

    # best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
    #
    # hyps = get_aux_labels(best_path)
    #
    # if params.method == "ctc-decoding":
    #     hyps = ["".join([token_sym_table[i] for i in ids]) for ids in hyps]
    # else:
    #     assert params.method == "1best", params.method
    #     hyps = [" ".join([word_sym_table[i] for i in ids]) for ids in hyps]

    results = {}
    # for i, hyp in enumerate(hyps):
    #     results[batch[i][0]] = hyp.replace("▁", " ").strip()
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.sample_rate = 16000
    args.subsampling_factor = 4
    args.feature_dim = 80
    args.num_classes = 500

    wave_list: List[Tuple[str, str]] = []
    if args.wav_scp is not None:
        assert os.path.isfile(
            args.wav_scp
        ), f"wav_scp not exists : {args.wav_scp}"
        assert (
            args.output_file is not None
        ), "You should provide output_file when using wav_scp"
        with open(args.wav_scp, "r") as f:
            for line in f:
                toks = line.strip().split()
                assert len(toks) == 2, toks
                if not os.path.isfile(toks[1]):
                    logging.warning(f"File {toks[1]} not exists, skipping.")
                    continue
                wave_list.append(toks)
    else:
        assert len(args.sound_files) > 0, "No wav_scp or waves provided."
        for i, f in enumerate(args.sound_files):
            if not os.path.isfile(f):
                logging.warning(f"File {f} not exists, skipping.")
                continue
            wave_list.append((i, f))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    args.device = device

    logging.info(f"params : {args}")

    logging.info("Creating model")
    model = torch.jit.load(args.nn_model)
    model = model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = args.sample_rate
    opts.mel_opts.num_bins = args.feature_dim

    fbank = kaldifeat.Fbank(opts)

    token_sym_table = None
    word_sym_table = None
    if args.method == "ctc-decoding":
        logging.info("Use CTC decoding")
        max_token_id = args.num_classes - 1
        decoding_graph = k2.ctc_topo(
            max_token=max_token_id,
            device=device,
        )
        token_sym_table = k2.SymbolTable.from_file(args.tokens)
    else:
        assert args.method == "1best", args.method
        logging.info(f"Loading HLG from {args.HLG}")
        decoding_graph = k2.Fsa.from_dict(
            torch.load(args.HLG, map_location="cpu")
        )
        decoding_graph = decoding_graph.to(device)
        word_sym_table = k2.SymbolTable.from_file(args.words_file)
    decoding_graph = k2.Fsa.from_fsas([decoding_graph])
    decoding_graph.lm_scores = decoding_graph.scores.clone()

    results = {}
    start = 0
    while start + args.batch_size <= len(wave_list):

        if start % 100 == 0:
            logging.info(f"Decoding progress: {start}/{len(wave_list)}.")

        res = decode_one_batch(
            params=args,
            batch=wave_list[start: start + args.batch_size],
            model=model,
            feature_extractor=fbank,
            decoding_graph=decoding_graph,
            token_sym_table=token_sym_table,
            word_sym_table=word_sym_table,
        )
        start += args.batch_size

        results.update(res)

    logging.info(f"results : {results}")

    # if args.wav_scp is not None:
    #     output_dir = os.path.dirname(args.output_file)
    #     if output_dir != "":
    #         os.makedirs(output_dir, exist_ok=True)
    #     with open(args.output_file, "w", encoding="utf-8") as f:
    #         for x in wave_list:
    #             f.write(x[0] + "\t" + results[x[0]] + "\n")
    #     logging.info(f"Decoding results are written to {args.output_file}")
    # else:
    #     s = "\n"
    #     logging.info(f"results : {results}")
    #     for x in wave_list:
    #         s += f"{x[1]}:\n{results[x[0]]}\n\n"
    #     logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
