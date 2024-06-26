TensorFlow implementation of Echo-State Network with WM (working memory) unit described in the paper ["A neurodynamical model for working memory"](https://pubmed.ncbi.nlm.nih.gov/21036537/) by R. Pascanu and H. Jaeger

[ESN_WM.py](ESN_WM.py) has the RNN cell implementation of the ESN w/ WM specifically, within the custom "call" function

[ESN_WM-usage.py](ESN_WM-usage.py) has basic tests based on switching sin functions to negative with memory of certain inputs. This demonstrates a basic difference between the ESN w/ WM and standard ESN, namely, that the output layer of the ESN can utilize, through the reservoir, the WM memory to transform the output, while the standard ESN cannot. Further, the standard ESN is not able to perform certain transformations on inputs throughout time, while the ESN w/ WM can. The two figures show the difference after 100 epochs between the ESN w/ WM and the standard ESN. 






