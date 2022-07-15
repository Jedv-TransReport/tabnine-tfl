---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://images.unsplash.com/photo-1606606767399-01e271823a2e?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2940&q=80
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true
# some information about the slides, markdown enabled
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# persist drawings in exports and build
drawings:
  persist: false
# use UnoCSS (experimental)
css: unocss
---

# What is TabNine?

---

# TabNine Brief Intro

##### Tabnine describes itself on its homepage as an: 
<br/>

> "AI Assistant For Software Developers to Code Faster With Whole-Line & Full-Function Code Completions"

Its aim being to increase productivity and take auto-completion to the next level either project, company or open source community wide.

Using machine learning and other predictive techniques, TabNine can eventually quite accurately suggest what your about to write and it gives you a variable (default: 5) set of suggestions in order of % likelihood your going to want to use it. 

##### Very Quick Example:

<img alt="Example of TabNine Autocompletion" src="https://imgs.search.brave.com/W6oOoSBdnj8Xs3thMGNjUe1WQRFdz2qkDfy5Un6Ufgg/rs:fit:800:391:1/g:ce/aHR0cHM6Ly90ZXJt/aW5hbHJvb3QuY29t/LmJyL2Fzc2V0cy9p/bWcvcHJvZ3JhbWFj/YW8vdGFibmluZS5n/aWY.gif" width="350" />

--- 

# Can we use it at Transreport right now? 
<div style="color:red;">
The short answer right now is: *no*, on pain of Codrut pizza death.

We are applying at the moment to have it all legally reviewed so for now please just try to not fall asleep during the presentation but *PLEASE* do not install TabNine, GitHub CoPilot or any similar autocompletion that is not locally trained only or based on editor plugins for languages such as Vue extension packs with templates until we have had it signed off by legal & the boys at the top (me).
</div>
---

# How does it achieve this? 

- "Deep TabNine" - OpenAI’s GPT-2 Machine Learning Model (Transformer architecture)
> Trained on nearly 2 million files from GitHub, Deep TabNine comes with pre-existing knowledge, instead of learning only from a user’s current project.
- Additionally, the model also refers to documentation written in natural language to infer function names, parameters, and return types.
- It is capable of using small clues that are difficult for a traditional tool to access. For instance, it understands that the return type of app.get_user() is assumed to be an object with setter methods and the return type of app.get_users() is assumed to be a list.
- It also can scan your local project to further learn your language & implementation style.
- It can even use company-wide or enterprise team accounts to share learning model across an internal team and it's projects.
- It is completely language agnostic aslong as it has some existing learning trained on that language.

---

# What is OpenAI's GPT-2 ML Model?

Let's take the Wikipedia intro paragraphs as a good starting point:

> Generative Pre-trained Transformer 2 (GPT-2) is an open-source artificial intelligence created by OpenAI in February 2019. GPT-2 translates text, answers questions, summarizes passages, and generates text output on a level that, while sometimes indistinguishable from that of humans, can become repetitive or nonsensical when generating long passages. It is a general-purpose learner; it was not specifically trained to do any of these tasks, and its ability to perform them is an extension of its general ability to accurately synthesize the next item in an arbitrary sequence. GPT-2 was created as a "direct scale-up" of OpenAI's 2018 GPT model, with a ten-fold increase in both its parameter count and the size of its training dataset.

> The GPT architecture implements a deep neural network, specifically a transformer model, which uses attention in place of previous recurrence- and convolution-based architectures. Attention mechanisms allow the model to selectively focus on segments of input text it predicts to be the most relevant. This model allows for greatly increased parallelization, and outperforms previous benchmarks for RNN/CNN/LSTM-based models.

> OpenAI released the complete version of the GPT-2 language model (with 1.5 billion parameters) in November 2019. GPT-2 was to be followed by the 175-billion-parameter GPT-3, revealed to the public in 2020 (whose source code has never been made available). Access to GPT-3 is provided exclusively through an API offered by Microsoft.

---

# What does any of this mean? 

- ML or "Machine Learning" uses statistical analysis and data science to create learning mathematical models to make predictions on a confidence level of 0.0 to 1.0 (0-100%)
- Neural Networks are an upgrade on the original ML ideas and take it a step further, actually creating artificial neurons like in our brains and then making them learn over time.
- GPT is the king for this type of processing based around predictions on language, so in our case the typing of programming code or documentation.
- Technology has now reached the point where it can learn and over time apply better and better predictions to what we might type next, what the next news headline might be and all sorts of amazing functionality. (Some of you may of seen some famous image-generation or image-decoding ML models in recent years especially the dream type one done by Google.)

--- 
layout: two-cols
---
# What does a Neural Network look like?

- You have your inputs = where the data goes in to be processed. 
- You have 1 or more hidden layers that carry our the internal ML model's processing. Similar to logic gates in processors but based off of artificial neurons instead. 
- The neural network then has outputs that give the results. 
- Yes for anyone in the know this is a major simplification but I wanted it to be friendly to begin with ;) 
- We started off with just ANN and RNN, RNN was used for language processing (Recurrant Neural Networks) 
- RNN struggled with its long term learning memory, so someone needed to come along with something better.

::right::

<img src="https://www.researchgate.net/profile/Mostafa-Rahimi-Jamnani-2/publication/340646492/figure/fig4/AS:901557319127041@1591959423839/The-structure-of-the-artificial-neural-network-model-used-in-this-study.png" width="500" style="margin-top:6rem;" />

--- 

# Attention Is All You Need

In 2017 some researchers from Cornell University published a paper titled 'Attention is all you need'.
This addressed many issues of Recurrant Neural Networks and suggested a different paradigm approach they donned the 'transformer' neural network.

>We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

<br/>

>One main difference is that the input sequence can be passed parallelly so that GPU can be used effectively and the speed of training can also be increased. It is also based on the multi-headed attention layer, so it easily overcomes the vanishing gradient issue (the previously mentioned memory issues). The paper applies the transformer to an NMT.

<br/>

>For example, in a translator made up of a simple RNN, we input our sequence or the sentence in a continuous manner, one word at a time, to generate word embeddings. As every word depends on the previous word, its hidden state acts accordingly, so we have to feed it in one step at a time. 

<br/>

>In a transformer, however, we can pass all the words of a sentence and determine the word embedding simultaneously.

---

# So how did all this enable TabNine to get so good? 

Neural networks are all about the layers inside, and with the transformer architecture they could parse words and their contexts and sentences, positioning, translation all in parallel rather than before having to represent them one at a time, now with NLP (Natural Language Processing) - words are usually represented by vectors or other data types as computers typically don't understand any spoken language or any that can be interpreted more than one way.

With OpenAI releasing their GPT-2 Transformer based architecture, TabNine used it as the first building block in putting together one of the best examples of NLP & Machine Learning Prediction that I have seen today. Just like the Mintlify project which is now doing the same GPT tactics but to document your code!

Also with the ability for it to train using millions of lines of existing GitHub code covering all the top projects and most used languages - it gives it an extra edge compared to if you just locally or company internally train it. 

Though it can yes be set so all your internal code remains secure, you can host your own TabNine code server for enterprise use and security compliance.

Also with companies such as AMD & Nvidia really pushing the bar so far now with parallel ML based processing, just in the years it has taken for the researchers to come up with better models, the hardware side has also never stopped progressing at an extraordinary rate especially for GPUs due to the 'crypto boom' (Someone else do a TFL on that, I hate it these days).

---

# Is GitHub CoPilot as good? 

#### No.
<br/>

# Is this my own opinion or a fact? 

#### Both.
<br/>

# Should I ever install it if we do have legal permission?

#### Depends on how much you like me. Plus mixing it with TabNine is a big no-no and we all know which is better. ;)
<br/>

--- 

# Other Applications of GPT-2/3 Transformer Models 

- Governments used them during the pandemic to automatically search for disinformation across the internet and social media platforms. 
- Translation between spoken languages it can work really well with, e.g. French to English, English to German.
- GPT-3 now out has greatly improved processing but has not reached the likes of the TabNine architecture yet, we are expecting even bigger results in the next 5 years from GPT-3

<br/>

# Other Applications of Machine Learning in General

- ML has now hit many industries in a massive way, especially ones like the medical & pharmaceutical industry and disease research. 
- Facial recognition systems in cities, roads or companies worldwide have also started using proper ML (though obviously non-GPT neural networks)
- Image creation, recognition and editing has had some recent examples that really show just what is possible in the next 10-20 years if this kind of technology keeps surpassing expectations in time of growth.

---

# Resources & References 

- TabNine - (https://www.tabnine.com/)
- Attention is all you need (Academic Paper) - (https://arxiv.org/abs/1706.03762)
- Transformer Neural Networks - (https://builtin.com/artificial-intelligence/transformer-neural-network)
- Language Models are Unsupervised Multitask Learners (Academic Paper) - (https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- OpenAI GPT-2 Example Source Code - (https://github.com/openai/gpt-2)
- OpenAI GPT-2 Wikipedia - (https://en.wikipedia.org/wiki/GPT-2)
- GitHub CoPilot - (https://github.com/features/copilot/)

### Finding this presentation and the links

I made this presentation using sli.dev (Thanks CY) and have publically hosted it on a Git repository
on my Transreport user, so if you want access to the links or slides, please grab a clone or have a look on my user's repository over here:

https://github.com/Jedv-TransReport/tabnine-tfl