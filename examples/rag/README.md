# RAG example sets

A small help-centre corpus for an invoicing app, labelled by hand for retrieval
and for grounding. The files are the inputs of the RAG tests
(`tests/test_rag_examples.py`, `tests/test_rag_demo.py`) and of the RAG demo,
`./examples/rag/demo.sh`.

| File | What it holds |
|---|---|
| `corpus.jsonl` | 18 documents, `{"id", "text"}` |
| `retrieval.jsonl` | 12 retrieval cases: a query, graded relevance labels and a reference answer |
| `rankings.jsonl` | the top 5 documents two retrievers return per query: `baseline` and `regressed` |
| `faithfulness-labels.jsonl` | 16 answers written from the baseline's top 3, labelled `faithful` or `unfaithful` |
| `faithfulness.toml` | the faithfulness judge to calibrate against those labels |
| `pipeline.toml`, `pipeline-next.toml` | the RAG demo's pipeline on the current retriever and on the change under review |
| `server.py` | the demo's fakes: both retrievers, the reader and the grader (standard library only) |
| `endpoints.toml`, `providers.toml`, `prices.toml` | operator config for the fakes on port 8766 |
| `demo.sh` | the demo: both pipelines, the RAG report for each, and the gate |

## Relevance labels

`expected.relevance` grades each judged document for its query:

* 3: answers the query on its own
* 2: substantially helps (q10: the password reset page for a locked account)
* 1: related, but does not answer it
* 0: judged and not relevant (a near miss that shares words with the query)

A document without a label is not relevant. The ranking metrics treat grade 1
and above as relevant (`min_grade = 1`) and nDCG uses the grades as gains.
`expected.answer` is the reference answer the answer metrics and
`context_recall` read.

The `regressed` rankings push the answering document down or out of the top 5
for most queries: mean recall@3 falls from 1.0 to 5/9 and MRR from 11/12 to
227/360. `tests/test_rag_examples.py` derives these and per-case values by hand.

## Faithfulness labels

Each item is a question, the three documents the answer was written from
(`input.contexts`), the answer (`output`) and the label. `expected.unsupported`
lists the sentences the labeller found unsupported or contradicted; an answer
is `faithful` when that list is empty. Nine answers are faithful and seven are
not, each with one invented or contradicted detail: a wrong number (f02, f04,
f12), a policy reversed (f10), or a fact no source states (f07, f13, f16).

    evalkit calibrate --judge examples/rag/faithfulness.toml examples/rag/faithfulness-labels.jsonl

prints Cohen's kappa between the judge and the labels; the faithfulness scorer
is trusted to gate at 0.6 or better.
