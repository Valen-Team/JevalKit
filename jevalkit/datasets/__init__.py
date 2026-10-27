"""Dataset registry; each benchmark owns preparation, scoring and metadata."""

from importlib import import_module
from pathlib import Path

from ..io import load_json, read_tsv, sha256
from ..paths import data_root


BENCHMARKS = {
    "VisualDecisionBench": {
        "family": "VisualDecisionBench",
        "modality": "image,video",
        "module": "visual_decision_bench",
    },
    "VisualDecisionBench_Image": {
        "family": "VisualDecisionBench",
        "modality": "image",
        "module": "visual_decision_bench",
    },
    "VisualDecisionBench_Video": {
        "family": "VisualDecisionBench",
        "modality": "video",
        "module": "visual_decision_bench",
    },
    "JevBench": {"family": "JevBench", "modality": "text", "module": "jevbench"},
    "DecisionIndex": {"family": "DecisionIndex", "modality": "text", "module": "decision_index"},
    "DecisionIndex_Text": {"family": "DecisionIndex", "modality": "text", "module": "decision_index_text", "edition": "0.4.1"},
    "DecisionIndex_Vision": {"family": "DecisionIndex", "modality": "image", "module": "decision_index_vision", "edition": "0.4.1"},
    "TypedDecisions": {"family": "TypedDecisions", "modality": "text", "module": "typed_decisions"},
    "DecisionBench": {"family": "DecisionBench", "modality": "text", "module": "decision_bench"},
    "ProceduralTypedDecisions": {
        "family": "ProceduralTypedDecisions",
        "modality": "text",
        "module": "procedural_typed_decisions",
    },
    "ZhDecisionBench": {"family": "ZhDecisionBench", "modality": "text", "module": "zh_decision_bench"},
    "JevAltBench": {"family": "JevAltBench", "modality": "text", "module": "jevalt_bench"},
}


def published_manifest(name):
    """Load the frozen TSV metadata owned by the selected benchmark."""
    module = BENCHMARKS.get(name, {}).get("module")
    if module:
        path = Path(__file__).parent / module / "manifest.json"
        for manifest in load_json(path)["datasets"]:
            if manifest["name"] == name:
                return manifest
    raise FileNotFoundError("No published dataset metadata or local manifest for: " + name)


def prepare_dataset(name, root=None, source=None, edition=None):
    module = import_module("." + BENCHMARKS[name]["module"], __name__)
    return module.prepare(root=root, source=source, edition=edition or BENCHMARKS[name].get("edition", "0.3"))


def official_report(dataset, records, predictions, output):
    module_name = BENCHMARKS.get(dataset.name, {}).get("module")
    if module_name:
        module = import_module("." + module_name, __name__)
        scorer = getattr(module, "evaluate", None)
        if scorer:
            return scorer(dataset, records, predictions, output)
    return None


def configure_model_options(dataset, model_name, options):
    """Apply a benchmark's explicit native runtime requirements before provenance is sealed."""
    module_name = BENCHMARKS.get(dataset.name, {}).get("module")
    if module_name:
        module = import_module("." + module_name, __name__)
        configure = getattr(module, "configure_model_options", None)
        if configure:
            return configure(model_name, dict(options))
    return dict(options)


class DecisionDataset:
    def __init__(self, name, root=None):
        self.name = name
        self.root = data_root(root)
        self.path = self.root / "datasets" / (name + ".tsv")
        self.manifest_path = self.path.with_suffix(".manifest.json")
        self.manifest = (
            load_json(self.manifest_path) if self.manifest_path.is_file() else published_manifest(name)
        )
        if self.manifest["name"] != name or sha256(self.path) != self.manifest["sha256"]:
            raise ValueError("Dataset identity or TSV checksum mismatch: " + name)
        self.media_root = (self.path.parent / self.manifest.get("media_root", ".")).resolve()

    def __iter__(self):
        return read_tsv(self.path)

    def verify(self, media=False):
        from ..models.media import state_parts

        records, questions = 0, 0
        for record in self:
            records += 1
            questions += len(record["request"]["questions"])
            if media:
                state_parts(record["request"]["state"], self.media_root)
                for asset in record.get("assets", []):
                    if asset.get("sha256"):
                        path = self.media_root / asset.get("path", asset.get("url", ""))
                        if sha256(path) != asset["sha256"]:
                            raise ValueError("Media checksum mismatch: " + str(path))
        if records != self.manifest["records"] or questions != self.manifest["questions"]:
            raise ValueError("Dataset coverage differs from manifest")
        return {
            "name": self.name,
            "records": records,
            "questions": questions,
            "sha256": self.manifest["sha256"],
            "media_verified": media,
        }


def build_dataset(name, root=None):
    return DecisionDataset(name, root)
