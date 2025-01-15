# Samples the https://huggingface.co/datasets/gamino/wiki_medical_terms dataset
# to a smaller size to be used as example for the agentic rag demo.

import datasets

def main():
    dataset = "gamino/wiki_medical_terms"
    dataset = datasets.load_dataset(dataset, split="train")
    
    # Take random 2000 rows from the dataset
    dataset = dataset.shuffle(seed=42).select(list(range(2000)))
    dataset.push_to_hub("jamesnatulan/small_wiki_medical_terms")


if __name__ == "__main__":
    main()
