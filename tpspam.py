import numpy as np
import os
import math

def lireMail(fichier, dictionnaire):
	""" 
	Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
	"""
	f = open(fichier, "r",encoding="ascii", errors="surrogateescape")
	mots = f.read().lower().split(" ")
	
	x = [False] * len(dictionnaire) 


	for i in range(len(dictionnaire)):
		if dictionnaire[i].lower() in mots:
			x[i] = True
	
	f.close()
	return x

def charge_dico(fichier):
	f = open(fichier, "r")
	mots = f.read().split("\n")
	f.close()

	# Exclure les mots de moins de 3 lettres
	mots = [m for m in mots if len(m) >= 3]

	print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")

	return mots

def apprendBinomial(dossier, fichiers, dictionnaire):
	"""
	Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier
	Retourne un vecteur b de paramètres 
		
	"""
	eps = 1
	n=len(fichiers) #nb de mails
	#nb de mail contenant le mot i du dictionnaire
	nbMail = np.zeros(len(dictionnaire))

	for fichier in fichiers:
		chemin = os.path.join(dossier, fichier)
		x = lireMail(chemin, dictionnaire)
		nbMail += x #
	# Proba
	b = (nbMail + eps )/(n + 2 * eps)
	return b


def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Prédit si un mail représenté par un vecteur booléen x est un spam
		à partir du modèle de paramètres Pspam, Pham, bspam, bham.
		Retourne True ou False.
		
	"""
	#pn essaie de calculer P(spam∣x) et P(ham∣x)
	# Mais on a besoin de P(x), P(x|Spam) et P(x|Ham)
	PXSpam = 1
	PXHam = 1
	for i in range(len(x)):
		if x[i] :
			PXSpam *= bspam[i]
			PXHam *= bham[i]
		else:
			PXSpam *= (1-bspam[i])
			PXHam *= (1-bham[i])

	#Px = P(x∣spam)*P(spam) + P(x∣ham)⋅P(ham)
	Px = PXSpam * Pspam + PXHam * Pham
	# on calcule P(spam∣x)
	PSpamX = (PXSpam * Pspam) / Px

	# on calcule P(ham∣x)
	PHamX = (PXHam * Pham) / Px
	
	return PSpamX > PHamX
	
def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	"""
		Test le classifieur de paramètres Pspam, Pham, bspam, bham 
		sur tous les fichiers d'un dossier étiquetés 
		comme SPAM si isSpam et HAM sinon
		
		Retourne le taux d'erreur 
	"""
	fichiers = os.listdir(dossier)
	nbErreurs = 0
	for fichier in fichiers:
		#print("Mail " + dossier+"/"+fichier)
		pred = prediction(lireMail(dossier+"/"+fichier, dictionnaire), Pspam, Pham, bspam, bham)
		texte = "SPAM" if isSpam else "HAM"
		texte += " "+ dossier+"/"+fichier + " identifié comme un "
		texte += "SPAM" if pred else "HAM"
		if (isSpam and not pred) or (not isSpam and pred) :
			texte += " *** erreur ***"
			nbErreurs += 1
		print(texte)


	PourcetageErreur = nbErreurs / len(fichiers)

	return PourcetageErreur,len(fichiers)


############ programme principal ############

dossier_spams = "spam/baseapp/spam"	# à vérifier
dossier_hams = "spam/baseapp/ham"

fichiersspams = os.listdir(dossier_spams)
fichiershams = os.listdir(dossier_hams)

mSpam = len(fichiersspams)
mHam = len(fichiershams)

# Chargement du dictionnaire:
dictionnaire = charge_dico("spam/dictionnaire1000en.txt")
print(dictionnaire)

# Apprentissage des bspam et bham:
print("apprentissage de bspam...")
bspam = apprendBinomial(dossier_spams, fichiersspams, dictionnaire)
print("apprentissage de bham...")
bham = apprendBinomial(dossier_hams, fichiershams, dictionnaire)

# Calcul des probabilités a priori Pspam et Pham:
Pspam = mSpam /(mSpam + mHam)
Pham = mHam /(mSpam + mHam)


# Calcul des erreurs avec la fonction test():
dossier_spams_test = "spam/basetest/spam"	# à vérifier
dossier_hams_test = "spam/basetest/ham"
ErreurSpam, nbMailsSpam = test(dossier_spams_test, True, Pspam, Pham, bspam, bham)
ErreurHam, nbMailsHam = test(dossier_hams_test, False, Pspam, Pham, bspam, bham)
erreur = (ErreurSpam + ErreurHam) / 2
nbMails = nbMailsSpam + nbMailsHam


print("Erreur de test sur "+str(nbMailsSpam)+ "SPAM       : "+ str(ErreurSpam)+" %")
print("Erreur de test sur "+str(nbMailsHam)+ "HAM        : "+ str(ErreurHam)+" %")
print("Erreur de test globale sur "+str(nbMails)+ "mails : "+ str(erreur)+" %")

