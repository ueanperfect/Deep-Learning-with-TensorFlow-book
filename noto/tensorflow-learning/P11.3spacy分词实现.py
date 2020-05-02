import spacy
import neuralcoref
nlp = spacy.load('en_core_web_sm')
##
txt = "A magnetic monopole is a hypothetical elementary particles."
doc = nlp(txt)
tokens = [token for token in doc]
print(tokens)
pos = [token.pos_ for token in doc]
print(pos)
# lem = [token.lemma_ for token in doc]
# print(lem)
# stop_words = [token.is_stop for token in doc]
# print(stop_words)
# dep = [token.dep_ for token in doc]
# print(dep)
# noun_chunks = [nc for nc in doc.noun_chunks]
# print(noun_chunks)
# ##
# txt = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
# doc = nlp(txt)
# # ners = [(ent.text, ent.label_) for ent in doc.ents]
# # print(ners)
# # ##
# # txt = "My sister has a son and she loves him."
# txt = "My sister has a son and she loves him."
# # 将预训练的神经网络指代消解加入到spacy的管道中
# from spacy import displacy
# displacy.render(doc, style='ent', jupyter=True)

# import textacy.extract
#
# nlp = spacy.load('en_core_web_sm')
#
# with open("/Users/faguangnanhai/Desktop/Magnetic monopole - Wikipedia.txt", "r") as fin:
#     txt = fin.read()
#
# doc = nlp(txt)
# statements = textacy.extract.semistructured_statements(doc, "monopole")
# for statement in statements:
#     subject, verb, fact = statement
#    print(f" - {fact}")