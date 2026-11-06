"""The Mycelium: an index of the corpora outside the conversation -- a repository, a folder of documents, the
onion's Cellar -- kept on disk in one encrypted store per corpus, that answers a question with cited source spans
(path, lines, the sha256 of their text) and says when a source changed since it was indexed. This first layer
cuts files by their syntax and keeps a full-text index and the table of what each part defines and names; it calls
no model."""
