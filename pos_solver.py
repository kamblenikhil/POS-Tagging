# Author: Nikhil Kamble, Anita Badrinarayanan and Dhruti Patel
#
# (Based on skeleton code by D. Crandall)
#

import random
import math

class Solver:
    pos_tags = ["noun", "adj", "adv", "adp", "conj", "det", "num", "pron", "prt", "verb", "x", "."]
    initial_probs = {}
    transition_count = {}
    transition_probs = {}
    emission_count = {}
    emission_probs = {}
    pos_tag_count = {}
    sl_transition_count = {}
    sl_transition_probs = {}
    emission_probabilities_gibbs = {}
    transition_probabilities_gibbs = {}
    prior_pos_prob = {}
    gibbs_emission_count = {}
    total_word_count = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            posterior_probability = 0
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
            # print(posterior_probability)
            return posterior_probability
        elif model == "HMM":
            return self.calc_HMM_posterior(sentence, label)
        elif model == "Complex":
            #return -999
            return self.calculate_complex_posterior(sentence,label)
        else:
            print("Unknown algo!")

    def calc_HMM_posterior(self, words, parts_of_speeches):
        initial_p = math.log(self.get_initial_probs(parts_of_speeches[0]), 10)
        emission_p = 0
        transition_p = 0
        for i in range(len(parts_of_speeches)):
            emission_p += math.log(self.get_emission_probs(words[i], parts_of_speeches[i]), 10)
            if i != 0:
                transition_p += math.log(self.get_transition_probs(parts_of_speeches[i - 1], parts_of_speeches[i]), 10)

        return initial_p + emission_p + transition_p

    def calculate_complex_posterior(self,sentence,label):
        words = list(sentence)
        posterior_probability = 0
    
        for i in range(len(label)):
            if i == len(sentence)-1:
                if words[i] in self.emission_probs.keys():
                    if label[i] in self.emission_probs[words[i]].keys():
                        ep = math.log(self.emission_probs[words[i]][label[i]],10)
                    else:
                        ep = math.log(0.0000001,10)
                else:
                    ep = math.log(0.0000001,10)
                    #ep = math.log(self.prior_pos_prob[pos_tag])
                posterior_probability = posterior_probability+ep
            elif i == len(sentence)-2:
                next_pos = label[i + 1]
                if words[i] in self.emission_probs.keys():
                    if label[i] in self.emission_probs[words[i]].keys():
                        ep = math.log(self.emission_probs[words[i]][label[i]],10)
                    else:
                        ep = math.log(0.0000001,10)
                        #ep = math.log(self.prior_pos_prob[pos_tag])
                else:
                    ep = math.log(0.0000001,10)
                    #ep = math.log(self.prior_pos_prob[pos_tag])
                if words[i] in self.emission_probabilities_gibbs.keys():
                    if label[i] in self.emission_probabilities_gibbs[words[i]].keys():
                        if next_pos in self.emission_probabilities_gibbs[words[i]][label[i]].keys():
                            epg = math.log(self.emission_probabilities_gibbs[words[i]][label[i]][next_pos],10)
                        else:
                            epg = math.log(0.0000001,10)
                            #epg = math.log(self.prior_pos_prob[pos_tag])
                    else:
                        epg = math.log(0.0000001,10)
                        #epg = math.log(self.prior_pos_prob[pos_tag])
                else:
                    epg = math.log(0.0000001,10)
                    #epg = math.log(self.prior_pos_prob[pos_tag])
                if label[i] in self.transition_probs.keys():
                    if next_pos in self.transition_probs[label[i]].keys():
                        tp = math.log(self.transition_probs[label[i]][next_pos],10)
                    else:
                        tp = math.log(0.0000001,10)
                        #tp = math.log(self.prior_pos_prob[pos_tag])
                else:
                    tp =math.log(0.0000001,10)
                    #tp = math.log(self.prior_pos_prob[pos_tag])
                posterior_probability = posterior_probability+ep+epg+tp
            else:
                next_pos = label[i + 1]
                next_to_next_pos = label[i + 2]
                if words[i] in self.emission_probs.keys():
                    if label[i] in self.emission_probs[words[i]].keys():
                        ep = math.log(self.emission_probs[words[i]][label[i]],10)
                    else:
                        ep = math.log(0.0000001,10)
                        #ep = math.log(self.prior_pos_prob[pos_tag])
                else:
                    ep = math.log(0.0000001,10)
                    #ep = math.log(self.prior_pos_prob[pos_tag])
                if words[i] in self.emission_probabilities_gibbs.keys():
                    if label[i] in self.emission_probabilities_gibbs[words[i]].keys():
                        if next_pos in self.emission_probabilities_gibbs[words[i]][label[i]].keys():
                            epg = math.log(self.emission_probabilities_gibbs[words[i]][label[i]][next_pos],10)
                        else:
                            epg = math.log(0.0000001,10)
                            #epg = math.log(self.prior_pos_prob[pos_tag])
                    else:
                        epg = math.log(0.0000001,10)
                        #epg = math.log(self.prior_pos_prob[pos_tag])
                else:
                    epg = math.log(0.0000001,10)
                    #epg = math.log(self.prior_pos_prob[pos_tag])
                if label[i] in self.transition_probs.keys():
                    if next_pos in self.transition_probs[label[i]].keys():
                        tp = math.log(self.transition_probs[label[i]][next_pos],10)
                    else:
                        tp = math.log(0.0000001,10)
                        #tp = math.log(self.prior_pos_prob[pos_tag])
                else:
                    tp = math.log(0.0000001,10)
                    #tp = math.log(self.prior_pos_prob[pos_tag])
                if label[i] in self.sl_transition_probs.keys():
                    if next_pos in self.sl_transition_probs[label[i]].keys():
                        if next_to_next_pos in self.sl_transition_probs[label[i]][next_pos].keys():
                            tp2 = math.log(self.sl_transition_probs[label[i]][next_pos][next_to_next_pos],10)
                        else:
                            tp2 = math.log(0.0000001,10)
                            #tp2 = math.log(self.prior_pos_prob[pos_tag])
                    else:
                        tp2 = math.log(0.0000001,10)
                        #tp2 = math.log(self.prior_pos_prob[pos_tag])
                else:
                    tp2 = math.log(0.0000001,10)
                    #tp2 = math.log(self.prior_pos_prob[pos_tag])
                posterior_probability = posterior_probability+ep+epg+tp+tp2
        return posterior_probability

    def calculate_prior_pos_probabilities(self,data):
        total_pos_tags = 0
        for i in range(0,len(data)):
            for j in range(0,len(data[i][1])):
                total_pos_tags = total_pos_tags+1
                if data[i][1][j] in self.prior_pos_prob.keys():
                    self.prior_pos_prob[data[i][1][j]] = self.prior_pos_prob[data[i][1][j]]+1
                else:
                    self.prior_pos_prob[data[i][1][j]] = 1
        for i in range(0,len(self.pos_tags)):
            if(self.pos_tags[i] in self.prior_pos_prob.keys()):
                self.prior_pos_prob[self.pos_tags[i].lower()] = self.prior_pos_prob[self.pos_tags[i].lower()]/total_pos_tags

    def build_initial_count(self, pos):
        if pos in self.initial_probs:
            self.initial_probs[pos] += 1
        else:
            self.initial_probs[pos] = 1

    def build_initial_probs(self, data):
        for pos in self.initial_probs:
            self.initial_probs[pos] = self.initial_probs[pos] / len(data)

    def build_emission_count(self, word, pos):
        if word not in self.emission_count:
            self.emission_count[word] = {pos: 1}
        elif pos not in self.emission_count[word]:
            self.emission_count[word][pos] = 1
        else:
            self.emission_count[word][pos] += 1

    def build_emission_probs(self):
        for word in self.emission_count:
            for pos in self.emission_count[word]:
                if word not in self.emission_probs:
                    self.emission_probs[word] = {pos: self.emission_count[word][pos] / self.pos_tag_count[pos]}
                else:
                    self.emission_probs[word][pos] = self.emission_count[word][pos] / self.pos_tag_count[pos]

    def build_gibbs_emission_count(self, word, pos, pos1):
        if word not in self.gibbs_emission_count:
            self.gibbs_emission_count[word] = {pos: {pos1: 1}}
        elif pos not in self.gibbs_emission_count[word]:
            self.gibbs_emission_count[word][pos] = {pos1:1}
        elif pos1 not in self.gibbs_emission_count[word][pos]:
            self.gibbs_emission_count[word][pos][pos1] = 1
        else:
            self.gibbs_emission_count[word][pos][pos1] += 1

    def calculate_emission_probab_gibbs(self):
        for word in self.gibbs_emission_count:
            for pos in self.gibbs_emission_count[word]:
                for pos1 in self.gibbs_emission_count[word][pos]:
                    if word not in self.emission_probabilities_gibbs:
                        self.emission_probabilities_gibbs[word] = {pos: {pos1:self.gibbs_emission_count[word][pos][pos1]/self.total_word_count[word]}}
                    elif pos not in self.emission_probabilities_gibbs[word]:
                        self.emission_probabilities_gibbs[word][pos] = {pos1:self.gibbs_emission_count[word][pos][pos1]/self.total_word_count[word]}
                    else:
                        self.emission_probabilities_gibbs[word][pos][pos1] = self.gibbs_emission_count[word][pos][pos1]/self.total_word_count[word]


    def build_transition_count(self,pos1,pos2):
        if pos1 not in self.transition_count:
            self.transition_count[pos1] = {pos2: 1}
        elif pos2 not in self.transition_count[pos1]:
            self.transition_count[pos1][pos2] = 1
        else:
            self.transition_count[pos1][pos2] += 1

    def build_second_level_transition_count(self,pos1,pos2,pos3):
        if pos1 not in self.sl_transition_count:
            self.sl_transition_count[pos1] = {pos2: {pos3:1}}
        elif pos2 not in self.sl_transition_count[pos1]:
            self.sl_transition_count[pos1][pos2] = {pos3:1}
        elif pos3 not in self.sl_transition_count[pos1][pos2]:
            self.sl_transition_count[pos1][pos2][pos3] = 1
        else:
            self.sl_transition_count[pos1][pos2][pos3] += 1

    def calculate_transition_probab(self):
        for pos1 in self.transition_count:
            for pos2 in self.transition_count[pos1]:
                if pos1 not in self.transition_probs:
                    self.transition_probs[pos1] = {pos2: self.transition_count[pos1][pos2] / self.pos_tag_count[pos1]}
                else:
                    self.transition_probs[pos1][pos2] = self.transition_count[pos1][pos2] / self.pos_tag_count[pos1]

    def calculate_second_level_transitions(self):
        for pos1 in self.sl_transition_count:
            for pos2 in self.sl_transition_count[pos1]:
                for pos3 in self.sl_transition_count[pos1][pos2]:
                    if pos1 not in self.sl_transition_probs:
                        self.sl_transition_probs[pos1] = {pos2:
                                                              {pos3:
                                                                   self.sl_transition_count[pos1][pos2][pos3]/self.pos_tag_count[pos1]}}
                    elif pos2 not in self.sl_transition_probs[pos1]:
                        self.sl_transition_probs[pos1][pos2] = {pos3:self.sl_transition_count[pos1][pos2][pos3]/self.pos_tag_count[pos1]}
                    else:
                        self.sl_transition_probs[pos1][pos2][pos3] = self.sl_transition_count[pos1][pos2][pos3] / self.pos_tag_count[pos1]


    def get_initial_probs(self, pos):
        if pos in self.initial_probs:
            return self.initial_probs[pos]
        return 0.0000001

    def get_emission_probs(self, word, pos):
        if word in self.emission_probs and pos in self.emission_probs[word]:
            return self.emission_probs[word][pos]
        return 0.0000001

    def get_transition_probs(self, pos1, pos2):
        if pos1 in self.transition_probs and pos2 in self.transition_probs[pos1]:
            return self.transition_probs[pos1][pos2]
        return 0.0000001

    def complex_mcmc(self,sentence):
        words = list(sentence)
        temp = {}
        prob = {}
        pos_tags = ['noun', 'det', 'adj', 'verb', 'adp', '.', 'adv', 'conj', 'pron', 'x', 'prt', 'num']
        bayes_tags = self.simplified(sentence)
        iterations = 0
        for i in range(len(sentence)):
            #temp[words[i]] = random.choice(pos_tags)
            temp[words[i]] = bayes_tags[i]
            #temp[words[i]] = 'noun'
        X = [temp]
        sample = dict(X[-1])
        while iterations < 200:
            for i in range(len(sentence)):
                for pos_tag in pos_tags:
                    if i == len(sentence)-1:
                        if words[i] in self.emission_probs.keys():
                            if pos_tag in self.emission_probs[words[i]].keys():
                                ep = math.log(self.emission_probs[words[i]][pos_tag],10)
                            else:
                                ep = 0.0000001
                                #ep = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            ep = 0.0000001
                            #ep = math.log(self.prior_pos_prob[pos_tag])
                        if words[i] in prob.keys():
                            if pos_tag in prob[words[i]].keys():
                                prob[words[i]][pos_tag] = prob[words[i]][pos_tag]+ep
                            else:
                                prob[words[i]][pos_tag] = ep
                        else:
                            prob[words[i]] = {pos_tag:ep}
                    elif i == len(sentence)-2:
                        next_pos = sample[words[i+1]]
                        if words[i] in self.emission_probs.keys():
                            if pos_tag in self.emission_probs[words[i]].keys():
                                ep = math.log(self.emission_probs[words[i]][pos_tag],10)
                            else:
                                ep = 0.0000001
                                #ep = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            ep = 0.0000001
                            #ep = math.log(self.prior_pos_prob[pos_tag])
                        if words[i] in self.emission_probabilities_gibbs.keys():
                            if pos_tag in self.emission_probabilities_gibbs[words[i]].keys():
                                if next_pos in self.emission_probabilities_gibbs[words[i]][pos_tag].keys():
                                    epg = math.log(self.emission_probabilities_gibbs[words[i]][pos_tag][next_pos],10)
                                else:
                                    epg = 0.0000001
                                    #epg = math.log(self.prior_pos_prob[pos_tag])
                            else:
                                epg = 0.0000001
                                #epg = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            epg = 0.0000001
                            #epg = math.log(self.prior_pos_prob[pos_tag])
                        if pos_tag in self.transition_probs.keys():
                            if next_pos in self.transition_probs[pos_tag].keys():
                                tp = math.log(self.transition_probs[pos_tag][next_pos],10)
                            else:
                                tp = 0.0000001
                                #tp = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            tp = 0.0000001
                            #tp = math.log(self.prior_pos_prob[pos_tag])
                        if words[i] in prob.keys():
                            if pos_tag in prob[words[i]].keys():
                                prob[words[i]][pos_tag] = prob[words[i]][pos_tag]+ep+epg+tp
                            else:
                                prob[words[i]][pos_tag] = ep+epg+tp
                        else:
                            prob[words[i]] = {pos_tag:ep+epg+tp}
                    else:
                        next_pos = sample[words[i + 1]]
                        next_to_next_pos = sample[words[i + 2]]
                        if words[i] in self.emission_probs.keys():
                            if pos_tag in self.emission_probs[words[i]].keys():
                                ep = math.log(self.emission_probs[words[i]][pos_tag],10)
                            else:
                                ep = 0.0000001
                                #ep = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            ep = 0.0000001
                            #ep = math.log(self.prior_pos_prob[pos_tag])
                        if words[i] in self.emission_probabilities_gibbs.keys():
                            if pos_tag in self.emission_probabilities_gibbs[words[i]].keys():
                                if next_pos in self.emission_probabilities_gibbs[words[i]][pos_tag].keys():
                                    epg = math.log(self.emission_probabilities_gibbs[words[i]][pos_tag][next_pos],10)
                                else:
                                    epg = 0.0000001
                                    #epg = math.log(self.prior_pos_prob[pos_tag])
                            else:
                                epg = 0.0000001
                                #epg = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            epg = 0.0000001
                            #epg = math.log(self.prior_pos_prob[pos_tag])
                        if pos_tag in self.transition_probs.keys():
                            if next_pos in self.transition_probs[pos_tag].keys():
                                tp = math.log(self.transition_probs[pos_tag][next_pos],10)
                            else:
                                tp = 0.0000001
                                #tp = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            tp = 0.0000001
                            #tp = math.log(self.prior_pos_prob[pos_tag])
                        if pos_tag in self.sl_transition_probs.keys():
                            if next_pos in self.sl_transition_probs[pos_tag].keys():
                                if next_to_next_pos in self.sl_transition_probs[pos_tag][next_pos].keys():
                                    tp2 = math.log(self.sl_transition_probs[pos_tag][next_pos][next_to_next_pos],10)
                                else:
                                    tp2 = 0.0000001
                                    #tp2 = math.log(self.prior_pos_prob[pos_tag])
                            else:
                                tp2 = 0.0000001
                                #tp2 = math.log(self.prior_pos_prob[pos_tag])
                        else:
                            tp2 = 0.0000001
                            #tp2 = math.log(self.prior_pos_prob[pos_tag])
                        if words[i] in prob.keys():
                            if pos_tag in prob[words[i]].keys():
                                prob[words[i]][pos_tag] = prob[words[i]][pos_tag]+ep+epg+tp+tp2
                            else:
                                prob[words[i]][pos_tag] = ep+epg+tp+tp2
                        else:
                            prob[words[i]] = {pos_tag:ep+epg+tp+tp2}
            sum_of_prob = sum(prob[words[i]].values())
            if(sum_of_prob == 0):
                sum_of_prob = 0.00000001
            for key in prob[words[i]].keys():
                prob[words[i]][key] = prob[words[i]][key]/sum_of_prob
                prob_sum = 0
                for probs in prob[words[i]].keys():
                    prob_sum = prob_sum+prob[words[i]][probs]
                    if random.random() < prob_sum:
                        sample[words[i]] = probs
                        break
            iterations = iterations+1
            if iterations > 20:
                X.append(sample)
        frequencies = {}
        for word in words:
            frequencies[word] = {}
            for sample in X:
                if word in sample.keys():
                    if word in frequencies.keys():
                        if sample[word] in frequencies[word].keys():
                            frequencies[word][sample[word]] = frequencies[word][sample[word]]+1
                        else:
                            frequencies[word][sample[word]] = 1
                    else:
                        frequencies[word] = {sample[word]:1}
        pos_tags_assigned = []
        for i in range(len(sentence)):
            keys = list(frequencies[words[i]].keys())
            values = list(frequencies[words[i]].values())
            pos_tags_assigned.append(keys[values.index(max(values))])
        return pos_tags_assigned

    # Do the training!
    #
    def train(self, data):
        self.calculate_prior_pos_probabilities(data)
        for sentence in data:
            first_pos = sentence[1][0]
            self.build_initial_count(first_pos)
            for i in range(len(sentence[0])):
                if sentence[1][i] in self.pos_tag_count:
                    self.pos_tag_count[sentence[1][i]] += 1
                else:
                    self.pos_tag_count[sentence[1][i]] = 1
                self.build_emission_count(sentence[0][i], sentence[1][i])
                if sentence[0][i] in self.total_word_count:
                    self.total_word_count[sentence[0][i]] += 1
                else:
                    self.total_word_count[sentence[0][i]] = 1
                if i > 0:
                    self.build_transition_count(sentence[1][i-1],sentence[1][i])
            for i in range(len(sentence[0]) - 2):
                self.build_gibbs_emission_count(sentence[0][i], sentence[1][i], sentence[1][i + 1])
                if i > 0:
                    self.build_second_level_transition_count(sentence[1][i - 1], sentence[1][i], sentence[1][i + 1])
        self.build_initial_probs(data)
        self.calculate_transition_probab()
        self.calculate_second_level_transitions()
        self.build_emission_probs()
        self.calculate_emission_probab_gibbs()
        pass

    # Functions for each algorithm. Right now this just returns nouns -- fix this!

    def simplified(self, sentence):
        pos_assigned = []
        words = list(sentence)
        for i in range(len(sentence)):
            p = []
            for pos in self.pos_tags:
                if words[i] in self.emission_probs.keys():
                    if pos in self.emission_probs[words[i]].keys():
                        p.append((self.emission_probs[words[i]][pos] * self.prior_pos_prob[pos], pos))
                    else:
                        p.append((0.000001, pos))
                else:
                    p.append((0.000001, pos))
            max = float('-inf')
            for i in range(len(p)):
                if max < p[i][0]:
                    index = i
                    max = p[i][0]
            pos_assigned.append(p[index][1])
        return pos_assigned

    def hmm_viterbi(self, sentence):
        # viterbi_table[0]['noun'] - here 0 is the 1st letter in the sentence
        viterbi_table = [{}]
        path_track = []
        path = []
        for unique_pos in self.pos_tags:
            viterbi_table[0][unique_pos] = self.get_initial_probs(unique_pos) * self.get_emission_probs(sentence[0],unique_pos)

        for i in range(1, len(sentence)):
            viterbi_table.append({})
            path_track.append({})
            for curr_pos in self.pos_tags:
                max_value = 0
                for pre_pos in self.pos_tags:
                    value = viterbi_table[i - 1][pre_pos] * self.get_transition_probs(pre_pos, curr_pos)
                    if max_value < value:
                        max_value = value
                        max_pos = pre_pos
                viterbi_table[i][curr_pos] = max_value * self.get_emission_probs(sentence[i], curr_pos)
                path_track[i - 1][curr_pos] = max_pos

        for unique_pos in self.pos_tags:
            max_last_level = 0
            if viterbi_table[len(sentence) - 1][unique_pos] > max_last_level:
                max_last_level_pos = unique_pos

        path.append(max_last_level_pos)

        for i in range(len(path_track) - 1, -1, -1):
            max_last_level_pos = path_track[i][max_last_level_pos]
            path.append(max_last_level_pos)

        return path[::-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")