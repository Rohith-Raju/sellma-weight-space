"""Swedish exam test bench.

The pipeline, in order:

    fetch    download past Högskoleprovet booklets and answer keys      (bench.fetch)
    extract  turn the PDFs into one record per question                 (bench.extract)
    split    freeze which exams are dev / test / gemma_extra            (bench.split)
    run      ask a model every question, run the damage checks          (bench.run)
    compare  compare a run against a baseline run                       (bench.report)

Every step reads and writes plain files under the project folder, so each
step can be inspected and rerun on its own.
"""
