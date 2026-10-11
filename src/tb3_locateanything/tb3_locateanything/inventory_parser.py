"""Strict labelled boxes from Eagle's native multi-category detection prompt.

Syntax validation only: labels are not verified visually and conflicting class
assignments remain observations for memory to resolve. Missing classes are not
treated as an explicit absence. No ROS, image library or model dependency.
"""
import re

if __package__:
    from .output_parser import parse_locateanything_output
else:  # model_worker.py is also executed directly by its isolated interpreter.
    from output_parser import parse_locateanything_output


REF = re.compile(r"<ref>\s*([^<>]+?)\s*</ref>", re.IGNORECASE)
BOX = re.compile(r"<box>.*?</box>", re.IGNORECASE | re.DOTALL)
END = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")

# More explicit object names reduce part-as-object confusion in the native
# inventory prompt. They are model vocabulary, not new Memory classes or IDs.
MODEL_CATEGORY_NAMES = {
    "forklift": "forklift truck",
    "storage rack": "storage shelving unit",
}
CANONICAL_CATEGORY_NAMES = {name: label for label, name in MODEL_CATEGORY_NAMES.items()}


def normalize_categories(categories):
    """Validate categories and resolve model vocabulary to canonical classes."""
    if not isinstance(categories, (list, tuple)) or not categories:
        raise ValueError("inventory categories must be a non-empty list")
    normalized = []
    for label in categories:
        if not isinstance(label, str) or any(c in label for c in "<>"):
            raise ValueError("inventory category must be plain non-empty text")
        label = " ".join(label.lower().replace("_", " ").split())
        if not label:
            raise ValueError("inventory category must be plain non-empty text")
        label = CANONICAL_CATEGORY_NAMES.get(label, label)
        if label in normalized:
            raise ValueError("duplicate inventory category")
        normalized.append(label)
    return normalized


def model_categories(categories):
    """Use explicit model names while retaining canonical category identities."""
    return [MODEL_CATEGORY_NAMES.get(label, label) for label in normalize_categories(categories)]


def _conflicts(observations):
    same_box = {}
    for item in observations:
        same_box.setdefault(tuple(item["box"]), set()).add(item["label"])
    return [{"box": list(box), "labels": sorted(labels)}
            for box, labels in same_box.items() if len(labels) > 1]


def parse_inventory_output(answer, width, height, categories):
    """Parse one or more ``<ref>label</ref><box>...</box>`` groups.

    A label may precede several boxes, or recur before each instance. Every box
    must belong to a requested label. ``None`` is per label, so one class can be
    absent while another has boxes. A malformed group rejects the whole answer.
    Pixel boxes and canonical labels match Memory's coordinate/label convention.
    A model alias and its canonical label refer to the same requested class;
    their boxes/absence declarations are validated together, never separately.
    """
    requested = normalize_categories(categories)
    if not isinstance(answer, str):
        raise ValueError("inventory answer must be text")
    body = END.sub("", answer.strip()).strip()
    if not body:
        raise ValueError("inventory answer is empty")
    grouped = {}
    pos = 0
    while pos < len(body):
        ref = REF.match(body, pos)
        if ref is None:
            raise ValueError("inventory requires a paired ref before every box group")
        label = normalize_categories([ref.group(1)])[0]
        if label not in requested:
            raise ValueError("inventory returned an unrequested label: " + label)
        pos = ref.end()
        boxes = []
        while pos < len(body):
            while pos < len(body) and body[pos].isspace():
                pos += 1
            match = BOX.match(body, pos)
            if match is None:
                break
            boxes.append(match.group())
            pos = match.end()
        if not boxes:
            raise ValueError("inventory ref has no paired boxes: " + label)
        grouped.setdefault(label, []).extend(boxes)

    observations, absent = [], []
    for label, boxes in grouped.items():
        parsed = parse_locateanything_output("".join(boxes), width, height)
        if not parsed["parse_success"]:
            raise ValueError("invalid inventory boxes for " + label + ": " +
                             ",".join(parsed["errors"]))
        canonical = label.replace(" ", "_")
        if parsed["parse_status"] == "explicit_none":
            absent.append(canonical)
        observations.extend({"label": canonical, "box": box}
                            for box in parsed["pixel_boxes"])

    return {"observations": observations, "absent_labels": absent,
            "unreported_labels": [label.replace(" ", "_") for label in requested
                                  if label not in grouped], "conflicts": _conflicts(observations)}


def parse_inventory_response(response, width, height, categories):
    """Merge bounded single-class supplements before any geometry filtering.

    Each supplement must name a configured category missing boxes in the primary
    answer. A category gets at most one supplement. Bare catalog boxes or paired
    refs for that exact category are accepted; malformed/unrequested answers
    reject the entire response. A later positive observation supersedes an
    explicit primary absence, while the caller retains both raw model answers.
    """
    if not isinstance(response, dict):
        raise ValueError("inventory response must be an object")
    result = parse_inventory_output(response.get("answer"), width, height, categories)
    requested = normalize_categories(categories)
    supplements = response.get("supplemental_inventory", [])
    if not isinstance(supplements, list):
        raise ValueError("supplemental inventory must be a list")
    primary_labels = {item["label"] for item in result["observations"]}
    seen = set()
    for supplement in supplements:
        if not isinstance(supplement, dict):
            raise ValueError("inventory supplement must be an object")
        category = normalize_categories([supplement.get("category")])[0]
        label = category.replace(" ", "_")
        if category not in requested:
            raise ValueError("inventory supplement has an unrequested category: " + category)
        if label in seen:
            raise ValueError("duplicate inventory supplement: " + category)
        if label in primary_labels:
            raise ValueError("primary inventory already has boxes for supplement: " + category)
        seen.add(label)
        answer = supplement.get("answer")
        if not isinstance(answer, str):
            raise ValueError("inventory supplement answer must be text")
        body = END.sub("", answer.strip()).strip()
        if BOX.match(body):
            body = f"<ref>{category}</ref>" + body
        parsed = parse_inventory_output(body, width, height, [category])
        result["observations"].extend(parsed["observations"])
        result["unreported_labels"] = [x for x in result["unreported_labels"] if x != label]
        result["absent_labels"] = [x for x in result["absent_labels"] if x != label]
        result["absent_labels"].extend(parsed["absent_labels"])
    result["conflicts"] = _conflicts(result["observations"])
    return result
