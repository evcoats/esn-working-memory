TensorFlow implementation of Echo-State Network with WM (working memory) unit described in the paper ["A neurodynamical model for working memory"](https://pubmed.ncbi.nlm.nih.gov/21036537/) by R. Pascanu and H. Jaeger

[ESN_WM.py](ESN_WM.py) has the RNN cell implementation of the ESN w/ WM specifically, within the custom "call" function

[ESN_WM-usage.py](ESN_WM-usage.py) has basic tests based on switching sin functions to negative with memory of certain inputs. This demonstrates a basic difference between the ESN w/ WM and standard ESN, namely, that the output layer of the ESN can utilize, through the reservoir, the WM memory to transform the output, while the standard ESN cannot. Further, the standard ESN is not able to perform certain transformations on inputs throughout time, while the ESN w/ WM can. The [two figures](/figures) show the difference after 100 epochs between the ESN w/ WM and the standard ESN. 

Experiments 1-3 are listed in their respective files.

[Experiment 1](experiment1.py) utilizes working memory to switch a sin curve after seeing a certain number of inputs in a second dimension

[Experiment 2](experiment2.py) utilizes working memory to predict a Markov chain (performs the best)

[Experiment 3](experiment3.py) utilizes EEG data from the Physionet [Auditory evoked potential EEG-Biometric dataset](https://physionet.org/content/auditory-eeg/1.0.0/Filtered_Data/#files-panel), switching the channel it reads from based on whether it has passed a threshold in the previous channel, utilizing the WM units

Current Results:
The WM unit performs 2 orders of magnitude better on experiment 2, but at this point shows little difference on experiment 3 and 1. 






