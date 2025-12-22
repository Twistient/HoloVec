Common VSA patterns and design recipes.

## Role-Filler Binding

The most common pattern: bind roles to values.

```python
from holovec import VSA

model = VSA.create('FHRR', dim=2048)

# Define roles
ROLE_name = model.random(seed=1)
ROLE_age = model.random(seed=2)
ROLE_city = model.random(seed=3)

# Define fillers
name_alice = model.random(seed=10)
age_30 = encoder.encode(30)
city_boston = model.random(seed=20)

# Create record
person = model.bundle([
    model.bind(ROLE_name, name_alice),
    model.bind(ROLE_age, age_30),
    model.bind(ROLE_city, city_boston)
])

# Query: what is the name?
result = model.unbind(person, ROLE_name)
# result ≈ name_alice
```

**Use for**: Records, frames, semantic structures

---

## Sequence Encoding

### Position Binding

Bind each item with its position:

```python
def encode_sequence(items, model):
    pos_base = model.random(seed=42)
    parts = []
    for i, item in enumerate(items):
        pos_vec = model.permute(pos_base, k=i)
        parts.append(model.bind(item, pos_vec))
    return model.bundle(parts)
```

### Query by Position

```python
def query_position(sequence, position, model):
    pos_base = model.random(seed=42)
    pos_vec = model.permute(pos_base, k=position)
    return model.unbind(sequence, pos_vec)
```

**Use for**: Lists, arrays, time series

---

## Set Operations

Bundles naturally represent sets:

```python
# Set union (bundling)
set_ab = model.bundle([a, b])

# Membership test
def is_member(item, set_vec, threshold=0.4):
    return model.similarity(item, set_vec) > threshold

# Set intersection (approximate)
def intersect(set1, set2, codebook, store):
    # Find items that are similar to both
    results = []
    for label in codebook:
        vec = codebook[label]
        sim1 = model.similarity(vec, set1)
        sim2 = model.similarity(vec, set2)
        if sim1 > 0.4 and sim2 > 0.4:
            results.append(label)
    return results
```

**Use for**: Categories, tags, feature sets

---

## Hierarchical Structures

Use nested binding for trees:

```python
# Tree: parent -> child
#        root
#       /    \
#     left   right

CHILD_LEFT = model.random(seed=1)
CHILD_RIGHT = model.random(seed=2)

root = model.random(seed=10)
left = model.random(seed=11)
right = model.random(seed=12)

# Encode tree
tree = model.bundle([
    root,
    model.bind(CHILD_LEFT, left),
    model.bind(CHILD_RIGHT, right)
])

# Navigate: get left child from tree
left_result = model.unbind(tree, CHILD_LEFT)
```

**Use for**: Parse trees, organizational hierarchies

---

## Analogical Reasoning

Classic pattern: A is to B as C is to ?

```python
# king - man + woman = queen
king = word_vectors["king"]
man = word_vectors["man"]
woman = word_vectors["woman"]

# Compute analogy
query = model.bundle([king, model.unbind(woman, man)])
# or more precisely for some models:
# query = model.bind(model.unbind(king, man), woman)

# Find nearest word
result = store.query(query, k=1)
print(result)  # hopefully "queen"
```

**Use for**: Semantic relationships, knowledge completion

---

## Prototype Learning

One-shot classification via prototypes:

```python
from holovec.retrieval import Codebook, ItemStore

def create_prototype(examples, model):
    """Bundle examples into class prototype."""
    return model.bundle([encode(ex) for ex in examples])

# Create prototypes from few examples
prototypes = {
    "cat": create_prototype(cat_images, model),
    "dog": create_prototype(dog_images, model)
}
proto_codebook = Codebook(prototypes, backend=model.backend)
store = ItemStore(model).fit(proto_codebook)

# Classify new instance
def classify(instance):
    vec = encode(instance)
    results = store.query(vec, k=1)
    return results[0][0]
```

**Use for**: Few-shot learning, classification

---

## Temporal Patterns

Encode time-varying signals:

```python
# State at each timestep
def encode_temporal(states, model):
    TIME_ROLE = model.random(seed=99)
    parts = []
    for t, state in enumerate(states):
        time_vec = model.permute(TIME_ROLE, k=t)
        parts.append(model.bind(state, time_vec))
    return model.bundle(parts)

# Query: what was state at time t?
def query_time(trajectory, t, model):
    TIME_ROLE = model.random(seed=99)
    time_vec = model.permute(TIME_ROLE, k=t)
    return model.unbind(trajectory, time_vec)
```

**Use for**: Trajectories, event sequences, logs

---

## Multi-Modal Fusion

Combine different data types:

```python
# Define modality roles
MODALITY_text = model.random(seed=1)
MODALITY_image = model.random(seed=2)
MODALITY_audio = model.random(seed=3)

def encode_multimodal(text_vec, image_vec, audio_vec):
    return model.bundle([
        model.bind(MODALITY_text, text_vec),
        model.bind(MODALITY_image, image_vec),
        model.bind(MODALITY_audio, audio_vec)
    ])

# Query by modality
def get_text_component(multimodal_vec):
    return model.unbind(multimodal_vec, MODALITY_text)
```

**Use for**: Multi-sensor systems, multimedia retrieval

---

## Context-Dependent Binding

Bind with context for disambiguation:

```python
# "bank" in different contexts
CONTEXT_finance = model.random(seed=1)
CONTEXT_river = model.random(seed=2)

word_bank = model.random(seed=10)

# Same word, different meanings
bank_finance = model.bind(word_bank, CONTEXT_finance)
bank_river = model.bind(word_bank, CONTEXT_river)

# These are dissimilar despite same base word
print(model.similarity(bank_finance, bank_river))  # Low
```

**Use for**: Word sense disambiguation, context-aware AI

---

## Error Correction

Add redundancy for noise tolerance:

```python
# Encode with repetition
def encode_robust(item, model, copies=3):
    parts = [model.permute(item, k=i) for i in range(copies)]
    return model.bundle(parts)

# Decode by querying each copy
def decode_robust(noisy, model, copies=3):
    votes = []
    for i in range(copies):
        recovered = model.unpermute(noisy, k=i)
        votes.append(recovered)
    return model.bundle(votes)  # Majority vote
```

**Use for**: Noisy channels, unreliable storage

---

## See Also

- [Quick Start](../getting-started/quick-start.md) — Basic operations
- [Tutorials](../guides/patterns.md) — Full examples
- [Core Concepts](../getting-started/core-concepts.md) — Operation definitions
