# Generating high-quanlity symbolic music using fine-grained discriminators
## Abstract

Existing symbolic music generation methods utilize a discriminator to distinguish generated music from human-composed one based on a global perception of the music. However, a piece of music usually contains two main components: pitch and rhythm, which cannot be well measured at the same time by one discriminator. In this work, we propose to decouple rhythm and pitch, and design a fine-grained multi-discriminator architecture to tackle this issue.  Specifically, we design a pitch discriminator and a rhythm  discriminator. Equipped with a pitch augmentation strategy,  the pitch discriminator discerns the pitch distribution and  variations presented by the generated samples. By contrast,  the rhythm discriminator enhanced with bar-level relative  positional encodings focuses on the progression and velocity variations of the notes. Such a design allows the generator  to be more clearly aware of which aspect should be adjusted  of the generated music, rendering a ease to mimic human-composed music. Experimental results on the POP909 benchmark demonstrate the favorable performance of the proposed  method compared to several state-of-the-art symbolic music  generation models in terms of both objective and subjective  metrics. 

## The theoretical foundation in the field of music perception




|ID| Condition      | Ground Truth |Ours |Music Transformer|WGAN|Theme Transformer|
|:-------------:|:------------------:|:------:|:-----:|:----:|:--:|:--:|
|907 | <audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio> | <audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio>|<audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio>|<audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio>|<audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio>|<audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio>|
| 899 | good and plenty   | nice         |a||
| 895 | good `oreos`      | hmm          |a||
| 879 | good `zoute` drop | yumm         |a||

<audio src="M5000039TCS1207fNn.mp3" controls title="Title"></audio>
