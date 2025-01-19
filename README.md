# Is Llama 3.2's truthfulness language-dependent?

A study examining how language affects Llama 3.2's responses to truth vs. loyalty ethical dilemmas.

## About

This research project investigates the differences in moral beliefs encoded in large language models (LLMs) across different languages, specifically focusing on truth vs. loyalty dilemmas. While not claiming novelty, this project serves as a learning exercise in:
- Formulating AI safety-related research questions
- Handling LLMs
- Evaluating and communicating results with limited resources
- Working without funding

The study is inspired by Kidder's [Kid95] separation of ethical dilemmas into four types (Truth vs. Loyalty, Individual vs. Community, Short-Term vs. Long-Term, and Justice vs. Mercy), focusing specifically on truth vs. loyalty dilemmas.

## Features

- Dataset of 10 ethical dilemmas across different domains:
  - Healthcare
  - Friendship
  - Sports
  - Law Enforcement
  - Education
  - Family
  - Military
  - Legal
  - Media
  - Community
- Multi-language support (English, Dutch, Portuguese)
- Statistical analysis with error calculations
- Visualization generation
- Raw data preservation with timestamps

## Requirements

Install dependencies using the provided requirements.txt:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- Python 3.x
- numpy
- pandas
- matplotlib
- ollama

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and the LLaMA 3.2 model is available locally

## Dataset Creation

The ethical dilemmas were created using the following process:
1. Generated initial scenarios using Claude 3.5 Sonnet in English
2. Translated to Dutch and Portuguese by Claude 3.5 Sonnet
3. Validated translations by native speakers
4. Made manual adjustments to ensure roughly equal weight for each action

## Usage

Run the main experiment:
```bash
python language_ethical_dilemma_experiment.py
```

The script will:
1. Present dilemmas to the model in three languages
2. Run 1000 iterations per dilemma per language
3. Collect and categorize responses as "A" (truth), "B" (loyalty), or "invalid"
4. Generate visualizations and statistical analyses

## Output Files

- `experiment_results_[timestamp]_[n_runs]runs_[language].json`: Raw experimental data
- `response_distribution.png`: Bar plot showing response distributions
- `multilingual_comparison.png`: Cross-language comparison visualization

## Known Limitations

As a small-scale learning project, this study has several limitations:
1. Prompt variations: Only one format tested
2. Dataset size: Limited to 10 dilemmas
3. Model selection: Only tested on Llama3.2 due to resource constraints
4. Language scope: Limited to European languages (Dutch and Portuguese)

## Future Work Possibilities

Potential extensions of this research include:
1. Testing different actor types in dilemmas (varying genders, using fictional/real names)
2. Including non-European languages
3. Expanding to other types of ethical dilemmas
4. Adding confidence levels to model responses
5. Testing on larger models like Claude 3.5 Sonnet or GPT-4

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your chosen license here]

## Authors

- Janice van Dam
- Francisco Ferreira da Silva

## References

- [Kid95] Rushworth M Kidder. How good people make tough choices. Morrow New York, 1995.
- [YMS24] Jiaqing Yuan, Pradeep K Murukannaiah, and Munindar P Singh. Right vs. right: Can llms make tough choices? arXiv preprint arXiv:2412.19926, 2024.
- [SSFB24] Nino Scherrer, Claudia Shi, Amir Feder, and David Blei. Evaluating the moral beliefs encoded in llms. Advances in Neural Information Processing Systems, 36, 2024.
