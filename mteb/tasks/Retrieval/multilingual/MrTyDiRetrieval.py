from __future__ import annotations

import datasets

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"

_LANGS = {
    "ara": ["ara-Arab"],
    "ben": ["ben-Beng"],
    "eng": ["eng-Latn"],
    "fin": ["fin-Latn"],
    "ind": ["ind-Latn"],
    "jpn": ["jpn-Jpan"],
    "kor": ["kor-Kore"],
    "rus": ["rus-Cyrl"],
    "swh": ["swh-Latn"],
    "tel": ["tel-Telu"],
    "tha": ["tha-Thai"],
}

_LANGS_MAP = {
    "ara": "arabic",
    "ben": "bengali",
    "eng": "english",
    "fin": "finnish",
    "ind": "indonesian",
    "jpn": "japanese",
    "kor": "korean",
    "rus": "russian",
    "swh": "swahili",
    "tel": "telugu",
    "tha": "thai",
}


class MrTyDiRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTyDiRetrieval",
        dataset={
            "path": "castorini/mr-tydi",
            "revision": "1d43c80218d06d0ef80f5b172ccabd848b948bc1",
        },
        description=(
            "Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations."
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        main_score="ndcg_at_10",
        license="apache-2.0",
        domains=["Web"],
        text_creation="created",
        date=("2019-01-01", "2021-12-08"),
        form=["written"],
        task_subtypes=[],
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        bibtex_citation="""
        @article{mrtydi,
            title={{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval}, 
            author={Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
            year={2021},
            journal={arXiv:2108.08787},
        }
        """,
        n_samples={_EVAL_SPLIT: 744},
        avg_character_length={_EVAL_SPLIT: 589.184},
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        langs = self.metadata.eval_langs

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        for lang in langs:
            mrtydi_lang = _LANGS_MAP[lang]
            query_list = datasets.load_dataset(
                name=mrtydi_lang, split=_EVAL_SPLIT, **self.metadata_dict["dataset"]
            )
            corpus_list = datasets.load_dataset(
                path="castorini/mr-tydi-corpus",
                name=mrtydi_lang,
                split="train",
                revision="3a3aa212bbe94a8cc0dc858710a3dad49d532054",
            )

            lang_queries = {}
            lang_relevant_docs = {}
            for row in query_list:
                lang_queries[row["query_id"]] = row["query"]
                lang_relevant_docs[row["query_id"]] = {
                    row["positive_passages"][0]["docid"]: 1
                }

            lang_corpus = {
                row["docid"]: {"title": row["title"], "text": row["text"]}
                for row in corpus_list
            }

            self.queries[lang] = {_EVAL_SPLIT: lang_queries}
            self.corpus[lang] = {_EVAL_SPLIT: lang_corpus}
            self.relevant_docs[lang] = {_EVAL_SPLIT: lang_relevant_docs}

        self.data_loaded = True
