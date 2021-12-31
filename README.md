# Part-of-Speech Tagging

We use a given part of speech list (POS Tags) to define the values for the emission, transition, and initial probability.<br>

<ul><li><b>Initial Probability</b> - The ratio of the number of times parts of speech appears in the first position to the total number of sentences is known as initial probability.<br><br>
</li><li><b>Transition Probability</b> - It is defined as the ratio of the probability of one parts of speech coming after another parts of speech divided by the probability of the occurence of that parts of speech. <br><br>
</li><li><b>Emission Probability</b> - It is the ratio of the probability of a word being a specific part of speech to the probability of the part of speech appearing in the entire training data.<br><br>
</li></ul>

<h3> 1. Simplified Bayes Net </h3> <br>
<ul><li>The initial probability of each parts of speech occurring in the dataset is multiplied by the parts of speech emission probability. Each parts of speech maximum value is added to the sentence's individual part of speech list. <br>
</li><li>We were facing challenges when any new word occurs which is not present in the training dataset. For such words we have set their probability to a very small value (0.0000001)</li></ul>
<br>

<b>SIMPLIFIED PROBABILITY CALCULATION -</b>

```PYTHON3
if pos in self.emission_probabilities[words[i]].keys():
    p.append((self.emission_probabilities[words[i]][pos]*self.prior_pos_prob[pos],pos))
else:
    p.append((0.000001,pos))
```
<b>SIMPLIFIED POSTERIOR PROBABILITY CALCULATION -</b>
```PYTHON3
for word in sentence:
  for pos in label:
    if word in self.emission_probs.keys():
      if pos in self.emission_probs[word].keys():
        posterior_probability = posterior_probability + math.log(self.emission_probs[word][pos],10) + \
                                  math.log(self.prior_pos_prob[pos],10)
      else:
        posterior_probability = posterior_probability + 0.0000001
    else:
      posterior_probability = posterior_probability + 0.0000001
return posterior_probability
```
<br>

<h3> 2. HMM (Viterbi)</h3>
_How the program works and Description_ - We implemented Viterbi to find the most probable sequence of POS tags for the each given sentence in test file. The training file was used to calculate the emission and transition probabilities. Emission probablities were the probability of the observed word in the sentence given the POS tag. And this was done for each POS tag for an observed word. Transition probabilities were the probability of a POS tag being the next in sequence given the previous POS tag. Initial probabilities were the probability that the POS tag occurs at the beginning of the sentence. We then used all this probabilities to get the viterbi table and then backtracked to find the best optimal sequence of POS tags for a given sentence. We then calculated posterior for HMM such that it's P(S(1)) {P(S(2)|S(1))P(S(3)|S(2))…P(S(n)|S(n-1))} {P(W(1)|S(1))…P(W(n)|S(n))} where S is the state which in our case is the identified POS tag and W is the word in the sentence. <br />
_Design Decisions_ - We used 3 different dictionaries to store emission, transition and initial probabilities. We also used list of dictionaries to store the values of viterbi table that are calculated using dynamic programming approach. Then we backtracked on the table and returned the list of sequence of POS tags for each sentence <br />

<br><br>

<h3> 3. Gibbs Sampling (MCMC) </h3> <br>
In MCMC Gibbs Sampling, random samples of the pos tags for every word in the sentences are created and based on the probabilities calculated based on the model below, the parts of speech for each word is determined.<br>
We have considered the first sample to be the output of the list of parts of speech tags for a sentence given by Bayes Net<br>
While selecting an initial sample, we tried the following 3 options:-
<ul><li>Parts of Speech tags for all the words were set to 'noun'</li>
<li>Parts of Speech tags were randomly assigned to the words from a list of POS Tags using random.choice()</li>
<li>Parts of Speech tags output of Simplified Bayes Net model were passed as a sample</li></ul>
It was found that from the above 3 choices, the output of Bayes Net gave the best accuracy.<br>

Once a sample is selected, we calculate the following probabilities for the same in order to determine a new and batter sample for the next iteration.<br>
<ul><li><b>Transition Probability</b> - It is defined as the ratio of the probability of one parts of speech coming after another parts of speech divided by the probability of the occurence of that parts of speech.</li>
<li><b>Emission Probability</b> - It is the ratio of the probability of a word being a specific part of speech to the probability of the part of speech appearing in the entire training data.</li>
<li><b>Second level Transition Probability</b> - In the complex model, we can see that there is a transition from the 1st word to the 3rd word, 2nd word to the 4th word and so on. This is the second level transition probability. We have calculated this as the number of occurances of the transition from pos tag 1 -> pos tag 2 -> pos tag 3 in the entire data divided by the number of occurances of pos tag 1 in the entire data.</li>
<li><b>Second level Emission Probability</b> - In the complex model, we can see that there is an emission from the 1st word to the 2nd pos tag, 2nd word to the 3rd pos tag and so on. This is the second level emission probability. We have calculated this as the number of occurances of the emssion from word 1 -> pos tag 1 and transition from pos tag 1-> pos tag 2 in the entire data divided by the number of occurances of word 1 in the entire data.</li></ul>
<br>

We have made the following assumptions for Gibbs Sampling:<br>
<ul><li>The log of the probabilities were taken as the probabilities were going beyond e^-256 and were not giving a good accuracy</li>
<li>The words and pos tags for which the probabilities were not calculated as they were not part of the training data, we have assumed the value to be 0.0000001.</li></ul><br>
    
We faced the following challenges while implementing Gibbs Sampling:<br>
<ul><li>The samples created were not giving a good accuracy and hence the log values for the probabilities had to be considered</li>
<li>We calculated the second level transition probability previously as the sum of the transition probability from pos tag 1 to pos tag2 and the transition probability from pos tag 2 to pos tag 3. This again led to a very low accuracy and we changed our approach.</li>
<li>Similar to second level transition probability, we calculated the second level emission probability as the sum of the emission probability from word 1 to pos tag 1 and the transition probability pos tag 1 to pos tag 2.</li>
<li>Another issue was that we were unable to get a good accuracy even after sampling and after having a discussion with Stephen, we removed the initial samples out of our entire samples before calculating the posterior probability.This helped in increasing the accuracy.</li></ul>

Below are the accuracies for each model and the sample posterior probablities for a sentence from the test file

![image](https://media.github.iu.edu/user/18146/files/f43ce100-52ed-11ec-9025-3441cc61f2ae)
![image](https://media.github.iu.edu/user/18146/files/0c146500-52ee-11ec-9fa7-463c78e426d4)
