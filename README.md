# Retrieval Augmented Text Generation

This repository contains Python code for performing retrieval-augmented text generation using the LangChain library and Huggingface checkpoints.

You can fact-check claims by considering a given CSV column that consists of abstracts of scientific articles.

## Prerequisites
- **requirements.txt**: Make sure to install the necessary Python packages and dependencies listed in the `requirements.txt` file.
- **Huggingface Token**: Access to Huggingface tokens is required to utilize pre-trained models effectively.
- **GPU with Less than 10 GB Usage**: Ensure your GPU has less than 10 GB of capacity for optimal performance.

With these prerequisites in place, you can explore retrieval-augmented text generation and confidently fact-check claims.

## Supported Claims
You can try fact-checking the following claims using the provided `abstract.csv` dataset, which contains 100 scientific articles related to the following claims:

1. Antarctic Peninsula is shrinking.
2. Burning fossil fuels like coal and oil is the main cause of greenhouse gas emissions.
3. Climate change affects tourism.
4. Climate change causes economic losses.
5. Human activities may cause global warming.
6. 5G networks contribute to the spread of COVID-19.
7. COVID-19 vaccines lead to infertility.
8. COVID-19 causes diabetes.
9. Vaccines cause autism.
10. Young people typically experience mild sickness from COVID-19.
11. Alcohol consumption may cause cancer.
12. Drinking alcohol kills brain cells.
13. Eating spicy food may cause stomach ulcers.
14. Masturbation is harmful or can cause health problems.
15. Drinking green tea is believed to burn fat.
16. Eating carrots can improve your eyesight.

## Usage Options
You have two options to explore and verify these claims:

1. Compare LLM Generated Answers: You can compare answers generated solely by LLM (Language Model) with answers generated using the retrieval-augmented LLM approach.

2. Access Source Documents: In the second option, you can also view the source documents that support the claims, providing additional context and credibility.

Feel free to customize the source documents or LLM IDs according to your preferences and research needs.

Explore, fact-check, and make informed conclusions using the provided resources.
