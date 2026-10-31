Place datasets here.

WELFake:
- file: welfake.csv
- columns: text,label (title optional)
- label may be 0/1 or Fake/Real; normalization is handled in code.

Evidence corpus:
- file: evidence.csv
- columns: evidence_id,text,source,label
- label: SUPPORT, CONTRADICT, or UNKNOWN when known.

Only use evidence that you have a legitimate right to use and cite its source in the paper.
