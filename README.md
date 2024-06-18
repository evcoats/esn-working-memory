TensorFlow implementation of Echo-State Network with WM (working memory) unit described in the paper ["A neurodynamical model for working memory"](https://pubmed.ncbi.nlm.nih.gov/21036537/) by R. Pascanu and H. Jaeger

[ESN_WM.py](ESN_WM.py) has the RNN cell implementation, specifically, within the custom "call" function

[ESN_WM-usage.py](ESN_WM-usage.py) has a couple of basic tests, based on switching sin functions to negative by remembering certain inputs to demonstrate difference between the ESN w/ WM and standard ESN

In the future, hoping to redo the implementation and tests from the paper
