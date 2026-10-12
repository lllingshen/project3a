# `tb3_interfaces`

Defines `SemanticQueryResult`, the message passed from LocateAnything to the
navigation adapter and coordinator. It carries the selected object ID, position,
reference frame, original command, and success or failure status.

The message fields are retained from Yiyuchen Hu's course interface. The
`detector_label` field carries LocateAnything's object class. This package
contains only the interface; it does not perform perception or query parsing.
