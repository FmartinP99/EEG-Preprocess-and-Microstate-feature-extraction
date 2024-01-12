import csv
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2
import dit


def __convert_sequence(seq, _from=None, _to=None):
    """
    Translates 1 sequence to another.
    @param _from: for example ["A", "B", "C", "D"]
    @param _to: for example [0,1,2,3]
    then if the incoming sequence is ["A","C","D","B"...], then the return will be [0,2,3,1,...]
    """
    if _from is None:
        _from = ["A", "B", "C", "D", "E"]
    if _to is None:
        _to = [0,1,2,3,4]


    convert_dict = {}
    for i in range(len(_from)):
        convert_dict[_from[i]] = _to[i]

    result = []
    for ind in seq:
        result.append(convert_dict[ind])

    return result

def _H_k(_X, ns, k):
    """Shannon's joint entropy from x[n+p:n-m]
    x: symbolic time series
    ns: number of symbols
    k: length of k-history
    """
    x = __convert_sequence(_X)
    N = len(x)
    f = np.zeros(tuple(k*[ns]))
    for t in range(N-k): f[tuple(x[t:t+k])] += 1.0
    f /= (N-k) # normalize distribution
    hk = -np.sum(f[f>0]*np.log(f[f>0]))
    #m = np.sum(f>0)
    #hk = hk + (m-1)/(2*N) # Miller-Madow bias correction
    return_dict = {f"Shannon_hk_{k}": hk}
    return return_dict


def _testMarkov0(_X, ns, alpha, verbose=True):
    """Test zero-order Markovianity of symbolic sequence x with ns symbols.
    Null hypothesis: zero-order MC (iid) <=>
    p(X[t]), p(X[t+1]) independent
    cf. Kullback, Technometrics (1962)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """
    X = __convert_sequence(_X)
    if verbose:
        print("ZERO-ORDER MARKOVIANITY:")
    n = len(X)
    f_ij = np.zeros((ns,ns))
    f_i = np.zeros(ns)
    f_j = np.zeros(ns)
    # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
    for t in range(n-1):
        i = X[t]
        j = X[t+1]
        f_ij[i,j] += 1.0
        f_i[i] += 1.0
        f_j[j] += 1.0
    T = 0.0 # statistic
    for i, j in np.ndindex(f_ij.shape):
        f = f_ij[i,j]*f_i[i]*f_j[j]
        if (f > 0):
            num_ = n*f_ij[i,j]
            den_ = f_i[i]*f_j[j]
            T += (f_ij[i,j] * np.log(num_/den_))
    T *= 2.0
    df = (ns-1.0) * (ns-1.0)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return_dict = {f"markov0_p": p}
    return return_dict


def _testMarkov1(_X, ns, alpha, verbose=True):
    """Test first-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """
    X = __convert_sequence(_X)
    if verbose:
        print("\nFIRST-ORDER MARKOVIANITY:")
    n = len(X)
    f_ijk = np.zeros((ns,ns,ns))
    f_ij = np.zeros((ns,ns))
    f_jk = np.zeros((ns,ns))
    f_j = np.zeros(ns)
    for t in range(n-2):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        f_ijk[i,j,k] += 1.0
        f_ij[i,j] += 1.0
        f_jk[j,k] += 1.0
        f_j[j] += 1.0

    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i][j][k]*f_j[j]*f_ij[i][j]*f_jk[j][k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0

    df = ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return_dict = {f"markov1_p": p}
    return return_dict


def _testMarkov2(_X, ns, alpha, verbose=True):
    """Test second-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t], X[t-1]) = p(X[t+1] | X[t], X[t-1], X[t-2])
    cf. Kullback, Technometrics (1962), Table 10.2.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """
    X = __convert_sequence(_X)
    if verbose:
        print("\nSECOND-ORDER MARKOVIANITY:")
    n = len(X)
    f_ijkl = np.zeros((ns,ns,ns,ns))
    f_ijk = np.zeros((ns,ns,ns))
    f_jkl = np.zeros((ns,ns,ns))
    f_jk = np.zeros((ns,ns))
    for t in range(n-3):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        l = X[t+3]
        f_ijkl[i,j,k,l] += 1.0
        f_ijk[i,j,k] += 1.0
        f_jkl[j,k,l] += 1.0
        f_jk[j,k] += 1.0
    T = 0.0
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        f = f_ijkl[i,j,k,l]*f_ijk[i,j,k]*f_jkl[j,k,l]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijkl[i,j,k,l]*f_jk[j,k]
            den_ = f_ijk[i,j,k]*f_jkl[j,k,l]
            T += (f_ijkl[i,j,k,l]*np.log(num_/den_))
    T *= 2.0
    df = ns*ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return_dict = {f"markov2_p": p}
    return return_dict


def _conditionalHomogeneityTest(_X, ns, l, alpha, verbose=True):
    """Test conditional homogeneity of non-overlapping blocks of
    length l of symbolic sequence X with ns symbols
    cf. Kullback, Technometrics (1962), Table 9.1.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        l: split x into non-overlapping blocks of size l
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
        ORIGINALLY THIS FUNCTION HAS BEEN IMPLEMENTED BY Frederic-vW at: https://github.com/Frederic-vW/eeg_microstates
    """

    X = __convert_sequence(_X)
    if verbose:
        print("\nCONDITIONAL HOMOGENEITY (three-way table):")
    n = len(X)
    r = int(np.floor(float(n)/float(l))) # number of blocks
    nl = r*l
    if verbose:
        print("Split data in r = {:d} blocks of length {:d}.".format(r,l))
    f_ijk = np.zeros((r,ns,ns))
    f_ij = np.zeros((r,ns))
    f_jk = np.zeros((ns,ns))
    f_i = np.zeros(r)
    f_j = np.zeros(ns)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in  range(r): # block index
        for ii in range(l-1): # pos. inside the current block
            j = X[i*l + ii]
            k = X[i*l + ii + 1]
            f_ijk[i,j,k] += 1.0
            f_ij[i,j] += 1.0
            f_jk[j,k] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0

    # conditional homogeneity (Markovianity stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i,j,k]*f_j[j]*f_ij[i,j]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = (r-1)*(ns-1)*ns
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return_dict = {f"homogeneity_p_{l}": p}
    return return_dict



def _symmetry_Test(X, ns, alpha, verbose=True):
    """Test symmetry of the transition matrix of symbolic sequence X with
       ns symbols
       cf. Kullback, Technometrics (1962)

       Args:
           x: symbolic sequence, symbols = [0, 1, 2, ...]
           ns: number of symbols
           alpha: significance level
       Returns:
           p: p-value of the Chi2 test for independence

       ORIGINALLY THIS FUNCTION HAS BEEN IMPLEMENTED BY Frederic-vW at: https://github.com/Frederic-vW/eeg_microstates
       """


    convert_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


    if verbose:
        print("\nSYMMETRY:")
    n = len(X)
    f_ij = np.zeros((ns, ns))
    for t in range(n - 1):
        i = convert_dict[X[t]]
        j = convert_dict[X[t + 1]]
        f_ij[i, j] += 1.0
    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i, j] * f_ij[j, i]
            if (f > 0):
                num_ = 2 * f_ij[i, j]
                den_ = f_ij[i, j] + f_ij[j, i]
                T += (f_ij[i, j] * np.log(num_ / den_))
    T *= 2.0
    df = ns * (ns - 1) / 2
    # p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")

    return_dict = {"symmetry_p": p}
    return return_dict


def calculate_matrix_transitions(X, ns):
    convert_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    n = len(X)
    f_ij = np.zeros((ns, ns))
    for t in range(n - 1):
        i = convert_dict[X[t]]
        j = convert_dict[X[t + 1]]
        f_ij[i, j] += 1.0

    return f_ij


def calculate_dits(keys, chances):
    #do dit calculation  in this function
    d = dit.Distribution(keys, chances)

    shannon_entropy = dit.shannon.entropy(d)
    print(f"Shannon entropy: {shannon_entropy}")

    extropy = dit.other.extropy(d)
    print(f"Extropy: {extropy}")


    prob_a = d['A']
    prob_b = d['B']
    prob_c = d['C']
    prob_d = d['D']
    prob_e = d['E']

    print(f"Probability of A: {prob_a}")
    print(f"Probability of B: {prob_b}")
    print(f"Probability of C: {prob_c}")
    print(f"Probability of D: {prob_d}")
    print(f"Probability of D: {prob_e}")

    return_dict = {}
    return_dict["prob_a"] = prob_a
    return_dict["prob_b"] = prob_b
    return_dict["prob_c"] = prob_c
    return_dict["prob_d"] = prob_d
    return_dict["prob_e"] = prob_e
    return_dict["dit_extropy"] = extropy
    return_dict["dit_shannon_entropy"] = shannon_entropy


    return return_dict



def make_sequence_from_transitions(classes, lengths):
    seq = []
    for idx, cl in enumerate(classes):
        for i in range(lengths[idx]):
            seq.append(cl)
    return seq



def calculate_transition_metrics(ids, categories, transitions, limit_from, limit_to, possible_switches, mode="", basedir_out=""):


    LIST_TO_WRITE = []
    LIST_TO_WRITE_homogeneity = []
    LIST_TO_WRITE_SHANNON = []

    headers = []
    headers_homm = []
    headers_shannon = []

    for ix, tr in enumerate(transitions):

        print(f"\nCalculating {ids[ix]}: \n")
        tr_array = tr.split(",")[:-1]
        classes = [t[0] for t in tr_array]
        lengths = [int(t[1:]) for t in tr_array]

        maximum_switches = len(tr_array)

        #####calculate all transitions according to the limits
        length_of_actual_switches = {}
        #fill the dictionary with the appropriate keys and the counter
        for class_switch in possible_switches:
            length_of_actual_switches[class_switch] = 0

        #calculate the transitions per class transition according to the limits   [limit_to, limit_from] provided above
        for class_switch in possible_switches:
            for i in range(len(classes) - 1):
                if classes[i] == class_switch[0] and classes[i + 1] == class_switch[1] \
                        and lengths[i] >= limit_from and lengths[i + 1] >= limit_to:
                    counter = length_of_actual_switches[class_switch]
                    length_of_actual_switches[class_switch] = counter + 1


        #####calculates all transitions without the limits
        length_of_maximum_switches = {}
        for class_switch in possible_switches:
            length_of_maximum_switches[class_switch] = 0
        for class_switch in possible_switches:
            for i in range(len(classes) - 1):
                if classes[i] == class_switch[0] and classes[i + 1] == class_switch[1]:
                    counter = length_of_maximum_switches[class_switch]
                    length_of_maximum_switches[class_switch] = counter + 1


        ######################################################
        # do further calculations here

        cl = make_sequence_from_transitions(classes,lengths)
        #matrix_transitions
        print(cl)

        matrix_tr = calculate_matrix_transitions(cl, 5)
        length_of_seq = len(cl)
        convert_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        transition_dict = {}
        print(matrix_tr)
        for i in range(5):
            for j in range(5):
                transition_dict[f"{convert_dict[i]}{convert_dict[j]}"] = matrix_tr[i, j] / length_of_seq
        print(transition_dict)


        ##symmetry matrix calculation

        symm_dict = _symmetry_Test(cl, 5, 30, verbose=True)

        print(f"Symmetry Matrix: {symm_dict}\n")

        # markos tests
        markov0_dict = _testMarkov0(cl, 5, 30)
        markov1_dict = _testMarkov1(cl, 5, 30)
        markov2_dict = _testMarkov2(cl, 5, 30)


        ###homogeneity matrix calculation
        homm_list = []
        homm_keys = []
        print(cl)
        for i in range(5, 15):
            homm_dict = _conditionalHomogeneityTest(cl, 5, i*5, 30, verbose=True)
            homm_list.append(homm_dict)
            homm_keys += homm_dict.keys()


        #Shannon joint entropy calculation
        shannon_list = []
        shannon_keys = []
        for i in range(1, 5):
            shannon_dict = _H_k(cl, 5, i)
            shannon_list.append(shannon_dict)
            shannon_keys += shannon_dict.keys()


        #dit calculation
        print("Calcualation dit characteristics: \n")
        c_a = 0
        c_b = 0
        c_c = 0
        c_d = 0
        c_e = 0
        length = 0
        for idx, _cl in enumerate(classes):
            length += lengths[idx]
            if _cl == "A":
                c_a += lengths[idx]
            if _cl == "B":
                c_b += lengths[idx]
            if _cl == "C":
                c_c += lengths[idx]
            if _cl == "D":
                c_d += lengths[idx]
            if _cl == "E":
                c_e += lengths[idx]

        chances = []
        chances.append(c_a/length)
        chances.append(c_b/length)
        chances.append(c_c/length)
        chances.append(c_d/length)
        chances.append(c_e/length)

        dit_dict = calculate_dits(["A","B","C","D","E"], chances)
        ##

        print("##############################################################################")



        dict_to_write = dict(symm_dict, **dit_dict)
        dict_to_write.update(dit_dict)
        dict_to_write = dict(dict_to_write, **markov0_dict)
        dict_to_write.update(markov0_dict)
        dict_to_write = dict(dict_to_write, **markov1_dict)
        dict_to_write.update(markov1_dict)
        dict_to_write = dict(dict_to_write, **markov2_dict)
        dict_to_write.update(markov2_dict)
        dict_to_write = dict(dict_to_write, **transition_dict)
        dict_to_write.update(transition_dict)
        dict_to_write["id"] = ids[ix]
        dict_to_write["category"] = categories[ix]
        LIST_TO_WRITE.append(dict_to_write)

        if not headers:
            headers = ['id', 'category']
            headers += symm_dict.keys()
            headers += dit_dict.keys()
            headers += markov0_dict.keys()
            headers += markov1_dict.keys()
            headers += markov2_dict.keys()
            headers += transition_dict.keys()

        #homogeneity dict
        dict_to_write_homm = {}
        for dic in homm_list:
            dict_to_write_homm = dict(dict_to_write_homm, **dic)
            dict_to_write_homm.update(dic)
        dict_to_write_homm["id"] = ids[ix]
        dict_to_write_homm["category"] = categories[ix]
        LIST_TO_WRITE_homogeneity.append(dict_to_write_homm)

        if not headers_homm:
            headers_homm = ['id', 'category']
            headers_homm += homm_keys

        #shannon dict
        dict_to_write_shannon = {}
        for dic in shannon_list:
            dict_to_write_shannon = dict(dict_to_write_shannon, **dic)
            dict_to_write_shannon.update(dic)
        dict_to_write_shannon["id"] = ids[ix]
        dict_to_write_shannon["category"] = categories[ix]
        LIST_TO_WRITE_SHANNON.append(dict_to_write_shannon)


        if not headers_shannon:
            headers_shannon = ['id', 'category']
            headers_shannon += shannon_keys


        print(LIST_TO_WRITE_homogeneity)

    Path(f"{basedir_out}/").mkdir(parents=True, exist_ok=True)
    with open(f"{basedir_out}/eeg_transition_metrics{mode}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE)

    with open(f"{basedir_out}/eeg_transition_homogeneity_metric{mode}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers_homm)
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE_homogeneity)

    with open(f"{basedir_out}/eeg_transition_shannon_metric{mode}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers_shannon)
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE_SHANNON)


def read_file_transitions(sequence_tranisitons_csv=None, outdir="characteristics"):
    df = pd.read_csv(sequence_tranisitons_csv)

    _transitions = df["transitions"].values.tolist()
    _ids = df["id"].values.tolist()
    _categories = df["category"].values.tolist()

    # from at least this long sequence transitions to
    limit_from = 1
    # at least this long sequence.
    limit_to = 1
    possible_switches = [
        "AB", "AC", "AD", "BC", "BD", "CD",
        "BA", "CA", "DA", "CB", "DB", "DC",
        "EA", "EB", "EC", "ED", "AE", "BE", "CE", "DE"]

    calculate_transition_metrics(_ids, _categories, _transitions, limit_from, limit_to, possible_switches, "",
                                 outdir)