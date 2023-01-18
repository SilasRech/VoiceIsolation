# Multi-Device Speech Enhancement for Privacy and Quality

Problem Scenario: There can be multiple speakers in one room which are speaking at the same time. This creates an overlapped speech signal, where one speaker is leaking into the conversation of the other speaker. This degrades the speech quality and means a privacy risk when confidential information is leaked to a different conversation.

<img src="https://www.dropbox.com/s/n3xf27ls7tavi3y/Scenario-1.jpg?raw=1" width="668px" height="301px">

## Key metrics of solution:


-   **3.7 mio.** network parameters offer real-time capability
    
-   PESQ score of **3.7** after attenuation
    
-   Listening tests confirm that the subjective speech quality is doubled (**MUSHRA score 75**)
    
-   No prior information is needed to identify the targeted speaker
    
-   Reduction of mutual information by **60%**

A multi-device setup can be used to isolate the dominant speaker by attenuating an undesired speaker. For this we use an adapted convolutional time-domain audio separation network. It uses two microphone inputs, 1. the mixed channel of the speaker to be isolated, 2. the mixed channel of the speaker that needs to be attenuated.

The network consists of two parts, one masking network, where a mask is generated for the undesired speaker. The inverse of that mask is then applied to the targeted speaker channel to remove the undesired speech content. An enhancement block is added to further increase the speech quality.

<img src="https://www.dropbox.com/s/odr5wa3lvs6c9zi/InverseMasking-1.png?raw=1" width="430px" height="430px">

Listening Example 1

[Listening Example for original speech mixture](https://www.dropbox.com/s/ovimudiwwi7i9zf/TestSeparation_CompleteScenario126.wav?dl=0)


[Listening Example for isolated speaker](https://www.dropbox.com/s/mrqoabirx2hqs71/TestSeparation126.wav?dl=0)


Listening Example 2

[Listening Example for original speech mixture](https://www.dropbox.com/s/t7ts8w0zei1srnv/TestSeparation_CompleteScenario0.wav?dl=0)


[Listening Example for isolated speaker](https://www.dropbox.com/s/nrihdfgsig6446p/TestSeparation0.wav?dl=0)


Listening Example 3

[Listening Example for original speech mixture](https://www.dropbox.com/s/66r77d0tfnf4cqc/TestSeparation_CompleteScenario4.wav?dl=0)


[Listening Example for isolated speaker](https://www.dropbox.com/s/e5v5eun9w928mg8/TestSeparation4.wav?dl=0)


