## Why Zettelkasten notes make ideal vector database inputs

Atomic notes map perfectly onto vector database chunks because each
note is already one self-contained idea. There is no need to split
the note further — the note IS the chunk.

Three properties make Zettelkasten notes retrieval-friendly. First,
atomicity: one idea per note means the embedding vector is sharp and
unambiguous, not an average of multiple topics. Second, self-contained
writing: because notes must be understandable without context, they
include enough surrounding explanation for the embedding model to
place them accurately in semantic space. Third, rich linking language:
the Related: section and cross-references add topically relevant
terminology that strengthens the vector signal.

In practice, a well-written permanent note with CHUNK_SIZE=400 rarely
needs to be split at all — it fits in one or two chunks naturally.

Related: [[atomic-note-principle]], [[what-is-zettelkasten]], [[linking-notes]]
