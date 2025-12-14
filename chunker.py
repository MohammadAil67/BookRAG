import os
os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\lib\poppler-25.12.0\Library\bin"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

#### INDEXING ####

docs = "Dakhil - 2018 - Class-(9-10) English For Today PDF Web.pdf"

pdftransformed = partition_pdf(
    filename=docs,
    extract_images=True,
    infer_table_structure=True,
    chunking_strategy="by_characters",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path="./Static",
)

print(pdftransformed)