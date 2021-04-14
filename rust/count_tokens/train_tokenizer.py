import argparse
import glob

import sentencepiece as spm
from sentencepiece.sentencepiece_model_pb2 import ModelProto


def train(args):
    # We arrange the token order such that the byte fallback tokens
    # <0x00>..<0xFF> are assigned their corresponding integer ids 0..255
    # No setting of the SentencePieceTrainer parameters will accomplish this
    # directly, because the control_symbols argument will insert the <mask>
    # token at the smallest index available, displacing a byte fallback token.
    # So we edit the trained model afterwards.
    # One consequence of this approach is that the pad token can no longer be
    # assigned to index 0, so we assign it another index instead. The pad token
    # needs a positive index (and not just a sentinel value of -1) because some
    # parts of the data pipeline represent tokens as *unsigned* 32-bit integers.
    unk_id, bos_id, eos_id, mask_id, pad_id = range(256, 256 + 5)
    spm.SentencePieceTrainer.train(
        input=args.input,
        input_format="tsv",
        model_prefix=args.model_prefix + ".intermediate",
        vocab_size=args.vocab_size - 1,  # Not counting <mask>
        model_type="unigram",
        byte_fallback=True,
        bos_id=bos_id,
        eos_id=eos_id,
        unk_id=unk_id,
        pad_id=min(mask_id, pad_id),  # Account for later inserting <mask>
        train_extremely_large_corpus=True,
        num_threads=92,
    )

    # Now edit the model

    with open(args.model_prefix + ".intermediate.model", "rb") as f:
        model = ModelProto.FromString(f.read())

    control = ModelProto.SentencePiece.Type.Value("CONTROL")
    model.pieces.insert(
        mask_id, ModelProto.SentencePiece(piece="<mask>", score=0.0, type=control)
    )

    # Patch the trainer_spec, just in case. Running with the patched trainer
    # spec won't re-create the correct token order, but it should at least
    # recreate the contents of the vocabulary
    model.trainer_spec.control_symbols.append("<mask>")
    model.trainer_spec.pad_id = pad_id
    model.trainer_spec.vocab_size = len(model.pieces)

    # Sanity check
    assert model.pieces[unk_id].piece == "<unk>"
    assert model.pieces[bos_id].piece == "<s>"
    assert model.pieces[eos_id].piece == "</s>"
    assert model.pieces[mask_id].piece == "<mask>"
    assert model.pieces[pad_id].piece == "<pad>"
    assert model.trainer_spec.bos_id == bos_id
    assert model.trainer_spec.eos_id == eos_id
    assert model.trainer_spec.unk_id == unk_id
    assert model.trainer_spec.pad_id == pad_id
    assert model.trainer_spec.vocab_size == len(model.pieces)

    if len(model.pieces) != args.vocab_size:
        print(
            "WARNING: generated vocab size",
            len(model.pieces),
            "does not match requested vocab size",
            args.vocab_size,
        )

    # Save patched model and vocabs
    model_path = args.model_prefix + ".model"
    print("Saving final model:", model_path)
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())

    vocab_path = args.model_prefix + ".vocab"
    print("Saving final vocabs:", vocab_path)
    with open(vocab_path, "w") as f:
        for piece in model.pieces:
            f.write(f"{piece.piece}\t{piece.score:.6g}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32768)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
