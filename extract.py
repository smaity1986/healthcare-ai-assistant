import zipfile

with zipfile.ZipFile(
    "chest-xray-pneumonia.zip", "r"
) as zip_ref:
    zip_ref.extractall("dataset")

print("Extraction completed")