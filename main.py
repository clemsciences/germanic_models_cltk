from naive_bayes import NaiveBayes


if __name__ == "__main__":
    nb = NaiveBayes(["cogito, ergo sum", "νόησις νοήσεως"], ["Latin", "Greek"])
    nb.get_likelihood()
    nb.prior()
    nb.likelihood()
    nb.probability_s()
    nb.predict()
