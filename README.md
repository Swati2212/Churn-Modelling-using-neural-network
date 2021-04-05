### Churn-Modelling-using-neural-network

Business Problem: Dataset of a bank with 10,000 customers measured lots of attributes of the customer and is seeing unusual churn rates at a high rate. Want to understand what the problem is, address the problem, and give them insights. 10,000 is a sample, millions of customer across Europe. Took a sample of 10,000 measured six months ago lots of factors (name, credit score, grography, age, tenure, balance, numOfProducts, credit card, active member, estimated salary, exited, etc.). For these 10,000 randomly selected customers and track which stayed or left.

**Goal**: create a geographic segmentation model to tell which of the customers are at highest risk of leaving.

Valuable to any customer-oriented organisations. Geographic Segmentation Modeling can be applied to millions of scenarios, very valuable. (doesn't have to be for banks, churn rate, etc.). Same scenario works for (e.g. should this person get a loan or not? Should this be approved for credit => binary outcome, model, more likely to be reliable). Fradulant transactions (which is more likely to be fradulant)

- Binary outcome with lots of independent variables you can build a proper robust model to tell you which factors influence the outcome.

### TECHNOLOGIES USED
The Code is written in Python 3.6.9 using google colaboratory. You can go to this link.
You can also use Jupyter Notebook. Touse JupyterNotebook, First, download Anaconda. By downloading Anaconda, you get conda, Python, Jupyter Notebook and hundreds of other open source packages. Now, to install Tensor flow and keras, follow steps below,
# install pip in the virtual environment
$ conda install pip
# install Tensorflow CPU version
$ pip install --upgrade tensorflow # for python 2.7
$ pip3 install --upgrade tensorflow # for python 3.*
# install Keras (Note: please install TensorFlow first)
$ pip install Keras
