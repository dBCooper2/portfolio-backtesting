# Portfolio Testing

This Project seeks to build a successful portfolio using financial analysis tools and benchmarking, and will be a place to practice backtesting on strategies

The project will involve creating 5 portfolios from a budget of $100,000 each. These portfolios will then be benchmarked using CAPM and the Fama-French Model to predict their expected returns. Portfolio data will be accessed using Yahoo Finance using the yfinance package and the [Yahoo Finance Scraper](https://github.com/dBCooper2/financial-modeling/blob/main/notebooks/data_processing/yahoo_finance_scraper.ipynb) I am writing if it is necessary.The rates needed to run these analyses will be pulled from Dr. [Kenneth French's Fama-French Dataset](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), which at the time of writing this only contains data up to 11-30-2023. After the analyses are run, the performance of each portfolio from the end date of the French Dataset to today will be calculated and compared using performance measures outlined in the [benchmarking notebook](https://github.com/dBCooper2/financial-modeling/blob/main/notebooks/data_processing/yahoo_finance_scraper.ipynb).

## Deliverables

- Notebooks:
  - 5 Sample Portfolios
  - CAPM Analysis of every Portfolio
  - Fama-French Analysis of Every Portfolio
  - Visualizing The Results of each Benchmark

- Scripts:
  - main.py: runs the analysis from the notebooks as a single file
  - analysis.py: a separate functions file for the analysis to make the code more readable, this will be broken up into multiple files if complexitiy increases
  - visualization.py: a file of functions to handle the visualization of the datasets, separated to make the code more readable

- Writeup:
  - A Document explaining the process of analyzing these portfolios and reflecting on what observations were made

- Repository Structure:
  - notebooks
    - sample_portfolios.ipynb
    - capm_analysis.ipynb
    - ff_analysis.ipynb
    - visualizations.ipynb
  - scripts
    - main.py
    - analysis.py
    - visualization.py
  - docs
    - writeup.md
    - data
      - csvs/excel/pickle files for financial data stored here

### Resources

[LaTeX Symbols](https://artofproblemsolving.com/wiki/index.php/LaTeX:Symbols)

[LaTeX Commands](https://artofproblemsolving.com/wiki/index.php/LaTeX:Commands#Subscripts_and_Superscripts)

[Creating a Table of Contents in Jupyter](https://medium.com/@ahmetekiz/creating-table-of-contents-in-jupyter-notebook-52a7c696817f#f730)
