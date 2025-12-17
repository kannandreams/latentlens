"""Preset datasets to demonstrate embedding differences."""

EXAMPLE_DATASETS = {
    "Synonyms (Semantic vs Literal)": [
        {"content": "The generated visuals were stunning.", "metadata": {"label": "synonym_pair_1"}},
        {"content": "The created graphics were beautiful.", "metadata": {"label": "synonym_pair_1"}},
        {"content": "The code was buggy.", "metadata": {"label": "synonym_pair_2"}},
        {"content": "The script had errors.", "metadata": {"label": "synonym_pair_2"}},
    ],
    "Negation (Sentiment)": [
        {"content": "I love this product.", "metadata": {"label": "positive"}},
        {"content": "I do not love this product.", "metadata": {"label": "negative_negated"}},
        {"content": "I hate this product.", "metadata": {"label": "negative_direct"}},
    ],
    "Polysemy (Double Meanings)": [
        {"content": "The crane lifted the heavy steel beam.", "metadata": {"label": "machine"}},
        {"content": "The crane waded in the shallow water.", "metadata": {"label": "bird"}},
        {"content": "He managed to crane his neck to see.", "metadata": {"label": "action"}},
    ],
    "Brand Associations (Knowledge)": [
        {"content": "Steve Jobs founded this company.", "metadata": {"label": "Apple"}},
        {"content": "The iPhone is a popular smartphone.", "metadata": {"label": "Apple"}},
        {"content": "Windows is a popular operating system.", "metadata": {"label": "Microsoft"}},
        {"content": "Bill Gates is a famous philanthropist.", "metadata": {"label": "Microsoft"}},
    ]
}
