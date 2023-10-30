# puzzle-knowledge-sdk

## What is this?

A Python package for interacting with Puzzle glo data and knowledge applications.

## Why is it important?

Enables developers to build knowledge applications and utilize expert models in sync with Large Language Models (LLMs). Puzzle glo data are an integral component to connecting expert decisions to generative language.

## Who is it for?

This is for developers of knowledge apps that utilize LLMs on their knowledge.

## How to use it?

### The transform method

Here are the inputs of the transform method
```py
GloLoader.transform:
  query: str                    ## the question
  documents: List[Document]     ## list of LangChain documents generated from GloLoader.load()
  header: str                   ## optional string header for prompt
  task: str                     ## optional string task for prompt
  rank_function: function       ## custom ranking function
  additional_args: dict         ## any additional parameters for rank function
```

### The customized rank function

Here are the general requirements of a rank function
- Minimum requirement of `documents`.
  - Technically, `query` isn't required, but is highly recommended for similarity related tasks.
- Optional inputs can be added if passed to `transform` in `additional_args`
- Perform some operations on the inputs. Possible operations can include but not limited to:
  - Ranking for similarity
  - Concept Selection
  - Cut-off for Context Window
- Return transformed contexts as a string

The output of a ranking function returns a string that will populate a `context` field in the `transform` method.

### How to craft a transform process

A transform process can be as complex or simple as needed, but should always transform glo data into a text prompt. A rank function enables the developer more freedom to define the process for their needs.

1. Plan a transformation
   1. The end result of a successful transform is a text prompt composed of three components:
      1. Header
      2. Context
      3. Task
2. Define a rank function
   1. The rank function will be responsible for constructing the "context" component
3. Pass all pertinent parameters to `transform` to perform the transformation

## Example Code

### Import the module
```py
from puzzle_knowledge_sdk.GloLoader import GloLoader
```

### Load glo data

Example glo data on Chinese Evergreen plant care
```py
glo_url = "https://raw.githubusercontent.com/wjonasreger/data/main/plant-glos/chinese-evergreen.json"
glo_loader = GloLoader(glo_url)
glo_documents = glo_loader.load(loadLinks=False)
```

*Note: `loadLinks` enables complete loading of links upon loading the glo data.*

### Transform glo data into prompt for LLM

Example question for the glo data
```py
question = "How often should I water this plant?"
```

1. Stack Concepts from top-bottom (no ranking)
```py
GloLoader.transform(
    query=question,
    documents=glo_documents,
    rank_function=None,
    additional_args={"max_width": 1024}
)
```

2. Rank by Concepts related to Query (Using only concepts)
```py
GloLoader.transform(
    query=question,
    documents=glo_documents,
    rank_function=GloLoader.rank_by_concepts,
    additional_args={"max_width": 1024}
)
```

3. Rank by Link Chunks related to Query (Using only links)
```py
GloLoader.transform(
    query=question,
    documents=glo_documents,
    header = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.",
    task = f"QUESTION: {question}\nANSWER: ",
    rank_function=GloLoader.rank_by_links,
    additional_args={"max_width": 1024, "chunk_size": 1024, "chunk_overlap": 128}
)
```

4. Rank by Concepts and Link Chunks related to Query (Rank by concepts, add links as additional contexts)
```py
GloLoader.transform(
    query=question,
    documents=glo_documents,
    rank_function=GloLoader.rank_by_concepts_and_links,
    additional_args={"max_width": 1024, "chunk_size": 1024, "chunk_overlap": 128}
)
```