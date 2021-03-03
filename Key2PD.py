from Bradykinesia import *
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import statistics 

class Key2PD():
    def __init__(self, classification = "UPDRS", selected = False, typist = False, brain = False, threeGroups = False):
        """
            Initialise class

            Parameters:
                classification(string): classification mode
                selected(boolean): option to select certain features
                typist(boolean): include typist information or not
                brain(boolean): select BRAIN test features only
                threeGroups(boolean): Exclude PD ON or not
        """
        self.subjects = Subjects().subjects
        self.selected = selected
        self.threeGroups = threeGroups
        self.typist = typist
        self.brain = brain
        if classification == "group":
            self.loadGroup()
        elif classification == "updrs10":
            self.loadUPDRS10()
        else:
            self.loadUPDRS()
        # self.params = {"dtc_updrs" : {"class_weight": "balanced", "criterion": "entropy", "max_depth": 8, "max_features": "log2", "min_samples_leaf": 0.2, "min_samples_split": 0.1, "splitter": "best"}, "lra_updrs" : {"C": 11.288378916846883, "class_weight": "balanced", "dual": False, "penalty": "l2", "solver": "newton-cg", "tol": 0.1}, "svc_updrs" : {"C": 1, "class_weight": "balanced", "gamma": 0.0001, "kernel": "rbf", "tol": 0.001}, "ann_updrs" : {"activation": "tanh", "alpha": 0.0001, "hidden_layer_sizes": (50, 100, 50), "learning_rate": "constant", "solver": "adam"}, "lra_group": {"C": 11.288378916846883, "class_weight": "balanced", "dual": False, "penalty": "l1", "solver": "liblinear", "tol": 1}, "dtc_group": {"class_weight": "balanced", "criterion": "entropy", "max_depth": 8, "max_features": "auto", "min_samples_leaf": 0.1, "min_samples_split": 0.2, "splitter": "best"}, "ann_group": {"activation": "tanh", "alpha": 0.0001, "hidden_layer_sizes": (100,), "learning_rate": "adaptive", "solver": "adam"}, "svc_group": {"C": 1, "class_weight": None, "gamma": 0.001, "kernel": "linear", "tol": 1}}

    def preprocess(self, df):
        """
            Preprocess features

            Parameters:
                df(dataframe): dataframe of features

            Output:
                dataframe: normalized dataframe
        """
        # normalize data
        contcols = [c for c in df.columns if c != "subject_id" and c != "UPDRS" and c != "diagnosis" and c != "UPDRS_dom" and c != "UPDRS_ndom" and c != "side" and c != "typist" and c != "hand"]
        df[contcols] = MinMaxScaler().fit_transform(df[contcols])
        df = df.fillna(0.0)
        if not self.typist:
            df = df.drop(columns=["typist"])
       
        return df
    
    def loadUPDRS(self):
        """
            Load all features from all subjects for classification

        """
        df = pd.DataFrame(columns=["subject_id", "hand", "m_vs", "mn_vs", "qp_vs", "m_se_ft", "mn_se_ft", "qp_se_ft", "m_se_dt", "mn_se_dt", "qp_se_dt", "m_KS", "mn_KS", "qp_KS", "m_slope_ft", "mn_slope_ft", "qp_slope_ft", "m_slope_dt", "mn_slope_dt", "qp_slope_dt", "m_std_error_ft", "mn_std_error_ft", "qp_std_error_ft", "m_std_error_dt", "mn_std_error_dt", "qp_std_error_dt", "m_intercept_ft", "mn_intercept_ft", "qp_intercept_ft", "m_intercept_dt", "mn_intercept_dt", "qp_intercept_dt", "qp_err", "mn_err", "m_err", "UPDRS", "typist", "qp_AT_30", "qp_DS_30", "qp_KS_30", "qp_IS_30", "qp_VS_30", "qp_AT_60", "qp_DS_60", "qp_IS_60", "qp_VS_60", "qp_hesitations_dt", "mn_hesitations_dt", "m_hesitations_dt", "qp_hesitations_ft", "mn_hesitations_ft", "m_hesitations_ft", "affected", "diagnosis", "mn_AT_60", "mn_DS_60", "mn_IS_60", "m_AT_60", "m_DS_60", "m_IS_60", "dominant"]) 
            
        for s in self.subjects:
            # M
            m_dom_slope_ft, m_dom_intercept_ft, _, _, m_dom_std_error_ft = stats.linregress(range(len(s.m_dom_ft)), s.m_dom_ft)
            m_dom_se_ft = getSequenceEffectScore(s.m_dom_ft)
            m_ndom_slope_ft, m_ndom_intercept_ft, _, _, m_ndom_std_error_ft = stats.linregress(range(len(s.m_ndom_ft)), s.m_ndom_ft)
            m_ndom_se_ft = getSequenceEffectScore(s.m_ndom_ft)
            
            m_dom_slope_dt, m_dom_intercept_dt, _, _, m_dom_std_error_dt = stats.linregress(range(len(s.m_dom_dt)), s.m_dom_dt)
            m_dom_se_dt = getSequenceEffectScore(s.m_dom_dt)
            m_ndom_slope_dt, m_ndom_intercept_dt, _, _, m_ndom_std_error_dt = stats.linregress(range(len(s.m_ndom_dt)), s.m_ndom_dt)
            m_ndom_se_dt = getSequenceEffectScore(s.m_ndom_dt)
            
            # MN
            mn_dom_slope_ft, mn_dom_intercept_ft, _, _, mn_dom_std_error_ft = stats.linregress(range(len(s.mn_dom_ft)), s.mn_dom_ft)
            mn_dom_se_ft = getSequenceEffectScore(s.mn_dom_ft)
            mn_ndom_slope_ft, mn_ndom_intercept_ft, _, _, mn_ndom_std_error_ft = stats.linregress(range(len(s.mn_ndom_ft)), s.mn_ndom_ft)
            mn_ndom_se_ft = getSequenceEffectScore(s.mn_ndom_ft)
            
            mn_dom_slope_dt, mn_dom_intercept_dt, _, _, mn_dom_std_error_dt = stats.linregress(range(len(s.mn_dom_dt)), s.mn_dom_dt)
            mn_dom_se_dt = getSequenceEffectScore(s.mn_dom_dt)
            mn_ndom_slope_dt, mn_ndom_intercept_dt, _, _, mn_ndom_std_error_dt = stats.linregress(range(len(s.mn_ndom_dt)), s.mn_ndom_dt)
            mn_ndom_se_dt = getSequenceEffectScore(s.mn_ndom_dt)
            
            # QP
            qp_dom_slope_ft, qp_dom_intercept_ft, _, _, qp_dom_std_error_ft = stats.linregress(range(len(s.qp_dom_ft)), s.qp_dom_ft)
            qp_dom_se_ft = getSequenceEffectScore(s.qp_dom_ft)
            qp_ndom_slope_ft, qp_ndom_intercept_ft, _, _, qp_ndom_std_error_ft = stats.linregress(range(len(s.qp_ndom_ft)), s.qp_ndom_ft)
            qp_ndom_se_ft = getSequenceEffectScore(s.qp_ndom_ft)
            
            qp_dom_slope_dt, qp_dom_intercept_dt, _, _, qp_dom_std_error_dt = stats.linregress(range(len(s.qp_dom_dt)), s.qp_dom_dt)
            qp_dom_se_dt = getSequenceEffectScore(s.qp_dom_dt)
            qp_ndom_slope_dt, qp_ndom_intercept_dt, _, _, qp_ndom_std_error_dt = stats.linregress(range(len(s.qp_ndom_dt)), s.qp_ndom_dt)
            qp_ndom_se_dt = getSequenceEffectScore(s.qp_ndom_dt)
            
            # Hasan 2019 Features
            qp_dom_IS_30 = statistics.variance(s.qp_dom_ft_30)
            qp_ndom_IS_30 = statistics.variance(s.qp_ndom_ft_30)
            qp_dom_IS_60 = statistics.variance(s.qp_dom_ft)
            qp_ndom_IS_60 = statistics.variance(s.qp_ndom_ft)
            mn_dom_IS_60 = statistics.variance(s.mn_dom_ft)
            mn_ndom_IS_60 = statistics.variance(s.mn_ndom_ft)
            m_dom_IS_60 = statistics.variance(s.m_dom_ft)
            m_ndom_IS_60 = statistics.variance(s.m_ndom_ft)
            
            # Right affected
            if (s.hand == "left"):
                if (s.side == 1):
                    affected_dom = False
                    affected_ndom = True
                # Left affected
                elif (s.side == 2):
                    affected_dom = True
                    affected_ndom = False
                # Both sides affected
                elif (s.side == 3):
                    affected_dom = True
                    affected_ndom = True
                # No information
                else:
                    affected_dom = False
                    affected_ndom = False
            else:
                if (s.side == 1):
                    affected_dom = True
                    affected_ndom = False
                # Left affected
                elif (s.side == 2):
                    affected_dom = False
                    affected_ndom = True
                # Both sides affected
                elif (s.side == 3):
                    affected_dom = True
                    affected_ndom = True
                # No information
                else:
                    affected_dom = False
                    affected_ndom = False
            
            # append dom hand
            df = df.append({"subject_id" : s.subject_id, "hand" : s.hand, "m_vs" : s.m_dom_vs, "mn_vs" : s.mn_dom_vs, "qp_vs" : s.qp_dom_vs, "m_se_ft" : m_dom_se_ft, "mn_se_ft" : mn_dom_se_ft, "qp_se_ft" : qp_dom_se_ft, "m_se_dt" : m_dom_se_dt, "mn_se_dt" : mn_dom_se_dt, "qp_se_dt" : qp_dom_se_dt, "m_KS" : len(s.m_dom_ft), "mn_KS" : len(s.mn_dom_ft), "qp_KS" : len(s.qp_dom_ft), "m_slope_ft" : m_dom_slope_ft, "mn_slope_ft" : mn_dom_slope_ft, "qp_slope_ft" : qp_dom_slope_ft, "m_slope_dt" : m_dom_slope_dt, "mn_slope_dt" : mn_dom_slope_dt, "qp_slope_dt" : qp_dom_slope_dt, "m_std_error_ft" : m_dom_std_error_ft, "mn_std_error_ft" : mn_dom_std_error_ft, "qp_std_error_ft" : qp_dom_std_error_ft, "m_std_error_dt" : m_dom_std_error_dt, "mn_std_error_dt" : mn_dom_std_error_dt, "qp_std_error_dt" : qp_dom_std_error_dt, "m_intercept_ft" : m_dom_intercept_ft, "mn_intercept_ft" : mn_dom_intercept_ft, "qp_intercept_ft" : qp_dom_intercept_ft, "m_intercept_dt" : m_dom_intercept_dt, "mn_intercept_dt" : mn_dom_intercept_dt, "qp_intercept_dt" : qp_dom_intercept_dt, "qp_err" : s.qp_dom_err, "mn_err" : s.mn_dom_err, "m_err" : s.m_dom_err, "diagnosis" : s.diagnosis, "UPDRS" : s.UPDRS_dom, "typist" : s.typist, "qp_AT_30" : np.mean(s.qp_dom_dt_30), "qp_DS_30" : s.qp_dom_DS_30, "qp_KS_30" : len(s.qp_dom_ft_30), "qp_IS_30" : qp_dom_IS_30, "qp_VS_30" : s.qp_dom_VS_30, "qp_AT_60" : np.mean(s.qp_dom_dt), "qp_DS_60" : s.qp_dom_DS_60, "qp_IS_60" : qp_dom_IS_60, "qp_VS_60" : s.qp_dom_VS_60, "m_hesitations_dt" : s.m_dom_out_dt, "mn_hesitations_dt" : s.mn_dom_out_dt, "qp_hesitations_dt" : s.qp_dom_out_dt, "m_hesitations_ft" : s.m_dom_out_ft, "mn_hesitations_ft" : s.mn_dom_out_ft, "qp_hesitations_ft" : s.qp_dom_out_ft, "affected" : affected_dom, "mn_AT_60" : np.mean(s.mn_dom_dt), "mn_DS_60" : s.mn_dom_DS_60, "mn_IS_60" : mn_dom_IS_60, "m_AT_60" : np.mean(s.m_dom_dt), "m_DS_60" : s.m_dom_DS_60, "m_IS_60" : m_dom_IS_60, "dominant" : True}, ignore_index=True) 
            # append ndom hand        
            df = df.append({"subject_id" : s.subject_id, "hand" : s.hand, "m_vs" : s.m_ndom_vs, "mn_vs" : s.mn_ndom_vs, "qp_vs" : s.qp_ndom_vs, "m_se_ft" : m_ndom_se_ft, "mn_se_ft" : mn_ndom_se_ft, "qp_se_ft" : qp_ndom_se_ft, "m_se_dt" : m_ndom_se_dt, "mn_se_dt" : mn_ndom_se_dt, "qp_se_dt" : qp_ndom_se_dt, "m_KS" : len(s.m_ndom_ft), "mn_KS" : len(s.mn_ndom_ft), "qp_KS" : len(s.qp_ndom_ft), "m_slope_ft" : m_ndom_slope_ft, "mn_slope_ft" : mn_ndom_slope_ft, "qp_slope_ft" : qp_ndom_slope_ft, "m_slope_dt" : m_ndom_slope_dt, "mn_slope_dt" : mn_ndom_slope_dt, "qp_slope_dt" : qp_ndom_slope_dt, "m_std_error_ft" : m_ndom_std_error_ft, "mn_std_error_ft" : mn_ndom_std_error_ft, "qp_std_error_ft" : qp_ndom_std_error_dt, "m_std_error_dt" : m_ndom_std_error_dt, "mn_std_error_dt" : mn_ndom_std_error_dt, "qp_std_error_dt" : qp_ndom_std_error_dt, "m_intercept_ft" : m_ndom_intercept_ft, "mn_intercept_ft" : mn_ndom_intercept_ft, "qp_intercept_ft" : qp_ndom_intercept_ft, "m_intercept_dt" : m_ndom_intercept_dt, "mn_intercept_dt" : mn_ndom_intercept_dt, "qp_intercept_dt" : qp_ndom_intercept_dt, "qp_err" : s.qp_ndom_err, "mn_err" : s.mn_ndom_err, "m_err" : s.m_ndom_err, "diagnosis" : s.diagnosis, "UPDRS" : s.UPDRS_ndom, "typist" : s.typist, "qp_AT_30" : np.mean(s.qp_ndom_dt_30), "qp_DS_30" : s.qp_ndom_DS_30, "qp_KS_30" : len(s.qp_ndom_ft_30), "qp_IS_30" : qp_ndom_IS_30, "qp_VS_30" : s.qp_ndom_VS_30, "qp_AT_60" : np.mean(s.qp_ndom_dt), "qp_DS_60" : s.qp_ndom_DS_60, "qp_IS_60" : qp_ndom_IS_60, "qp_VS_60" : s.qp_ndom_VS_60, "m_hesitations_dt" : s.m_ndom_out_dt, "mn_hesitations_dt" : s.mn_ndom_out_dt, "qp_hesitations_dt" : s.qp_ndom_out_dt, "m_hesitations_ft" : s.m_ndom_out_ft, "mn_hesitations_ft" : s.mn_ndom_out_ft, "qp_hesitations_ft" : s.qp_ndom_out_ft, "affected" : affected_ndom, "mn_AT_60" : np.mean(s.mn_ndom_dt), "mn_DS_60" : s.mn_ndom_DS_60, "mn_IS_60" : mn_ndom_IS_60, "m_AT_60" : np.mean(s.m_ndom_dt), "m_DS_60" : s.m_ndom_DS_60, "m_IS_60" : m_ndom_IS_60, "dominant" : False}, ignore_index=True)
        
        self.df_raw = df.copy(deep=True)
        self.df = self.preprocess(df)
        if not self.brain:
            self.df = df.drop(columns=["qp_AT_30", "qp_DS_30", "qp_KS_30", "qp_IS_30", "qp_VS_30"])
        if self.selected:
            self.X = self.df[["m_vs", "mn_vs", "qp_vs", "mn_taps", "qp_slope_ft", "m_slope_dt", "mn_slope_dt", "qp_slope_dt", "mn_std_error_ft", "qp_std_error_dt", "m_intercept_dt", "mn_intercept_dt", "qp_err", "mn_err", "m_err"]].to_numpy()
        else:
            self.X = self.df.drop(columns=["subject_id", "diagnosis", "UPDRS", "hand", "affected", "diagnosis", "dominant"]).to_numpy()
        self.y = self.df["UPDRS"].to_numpy()
        self.label_dict = dict(zip([0, 1, 2, 3], ["UPDRS_0", "UPDRS_1", "UPDRS_2","UPDRS_3"]))
        
    def loadUPDRS10(self):
        """
            Load all first 10 seconds features from all subjects for classification (used for experiment)

        """
        df = pd.DataFrame(columns=["subject_id", "hand", "m_se_ft", "mn_se_ft", "qp_se_ft", "m_se_dt", "mn_se_dt", "qp_se_dt", "m_KS", "mn_KS", "qp_KS", "m_slope_ft", "mn_slope_ft", "qp_slope_ft", "m_slope_dt", "mn_slope_dt", "qp_slope_dt", "m_std_error_ft", "mn_std_error_ft", "qp_std_error_ft", "m_std_error_dt", "mn_std_error_dt", "qp_std_error_dt", "m_intercept_ft", "mn_intercept_ft", "qp_intercept_ft", "m_intercept_dt", "mn_intercept_dt", "qp_intercept_dt", "qp_err", "mn_err", "m_err", "diagnosis", "UPDRS", "typist", "qp_AT", "qp_DS", "qp_IS", "qp_VS"])
            
        for s in self.subjects:
            if (s.m_dom_ft_10 == [] or s.m_ndom_ft_10 == [] or s.mn_dom_ft_10 == [] or s.mn_ndom_ft_10 == [] or s.qp_dom_ft_10 == [] or s.qp_ndom_ft_10 == [] or s.m_dom_dt_10 == [] or s.m_ndom_dt_10 == [] or s.mn_dom_dt_10 == [] or s.mn_ndom_dt_10 == [] or s.qp_dom_dt_10 == [] or s.qp_ndom_dt_10 == []):
                continue
            else:
                # M
                m_dom_slope_ft, m_dom_intercept_ft, _, _, m_dom_std_error_ft = stats.linregress(range(len(s.m_dom_ft_10)), s.m_dom_ft_10)
                m_dom_se_ft = getSequenceEffectScore(s.m_dom_ft_10)
                m_ndom_slope_ft, m_ndom_intercept_ft, _, _, m_ndom_std_error_ft = stats.linregress(range(len(s.m_ndom_ft_10)), s.m_ndom_ft_10)
                m_ndom_se_ft = getSequenceEffectScore(s.m_ndom_ft_10)

                m_dom_slope_dt, m_dom_intercept_dt, _, _, m_dom_std_error_dt = stats.linregress(range(len(s.m_dom_dt_10)), s.m_dom_dt_10)
                m_dom_se_dt = getSequenceEffectScore(s.m_dom_dt_10)
                m_ndom_slope_dt, m_ndom_intercept_dt, _, _, m_ndom_std_error_dt = stats.linregress(range(len(s.m_ndom_dt_10)), s.m_ndom_dt_10)
                m_ndom_se_dt = getSequenceEffectScore(s.m_ndom_dt_10)

                # MN
                mn_dom_slope_ft, mn_dom_intercept_ft, _, _, mn_dom_std_error_ft = stats.linregress(range(len(s.mn_dom_ft_10)), s.mn_dom_ft_10)
                mn_dom_se_ft = getSequenceEffectScore(s.mn_dom_ft_10)
                mn_ndom_slope_ft, mn_ndom_intercept_ft, _, _, mn_ndom_std_error_ft = stats.linregress(range(len(s.mn_ndom_ft_10)), s.mn_ndom_ft_10)
                mn_ndom_se_ft = getSequenceEffectScore(s.mn_ndom_ft_10)

                mn_dom_slope_dt, mn_dom_intercept_dt, _, _, mn_dom_std_error_dt = stats.linregress(range(len(s.mn_dom_dt_10)), s.mn_dom_dt_10)
                mn_dom_se_dt = getSequenceEffectScore(s.mn_dom_dt_10)
                mn_ndom_slope_dt, mn_ndom_intercept_dt, _, _, mn_ndom_std_error_dt = stats.linregress(range(len(s.mn_ndom_dt_10)), s.mn_ndom_dt_10)
                mn_ndom_se_dt = getSequenceEffectScore(s.mn_ndom_dt_10)


                # QP
                qp_dom_slope_ft, qp_dom_intercept_ft, _, _, qp_dom_std_error_ft = stats.linregress(range(len(s.qp_dom_ft_10)), s.qp_dom_ft_10)
                qp_dom_se_ft = getSequenceEffectScore(s.qp_dom_ft_10)
                qp_ndom_slope_ft, qp_ndom_intercept_ft, _, _, qp_ndom_std_error_ft = stats.linregress(range(len(s.qp_ndom_ft_10)), s.qp_ndom_ft_10)
                qp_ndom_se_ft = getSequenceEffectScore(s.qp_ndom_ft_10)

                qp_dom_slope_dt, qp_dom_intercept_dt, _, _, qp_dom_std_error_dt = stats.linregress(range(len(s.qp_dom_dt_10)), s.qp_dom_dt_10)
                qp_dom_se_dt = getSequenceEffectScore(s.qp_dom_dt_10)
                qp_ndom_slope_dt, qp_ndom_intercept_dt, _, _, qp_ndom_std_error_dt = stats.linregress(range(len(s.qp_ndom_dt_10)), s.qp_ndom_dt_10)
                qp_ndom_se_dt = getSequenceEffectScore(s.qp_ndom_dt_10)
                
                # Hasan 2019 Features
                qp_dom_IS_10 = statistics.variance(s.qp_dom_ft_10)
                qp_ndom_IS_10 = statistics.variance(s.qp_ndom_ft_10)

                # append dom hand
                df = df.append({"subject_id" : s.subject_id, "hand" : s.hand, "m_se_ft" : m_dom_se_ft, "mn_se_ft" : mn_dom_se_ft, "qp_se_ft" : qp_dom_se_ft, "m_se_dt" : m_dom_se_dt, "mn_se_dt" : mn_dom_se_dt, "qp_se_dt" : qp_dom_se_dt, "m_KS" : len(s.m_dom_ft_10), "mn_KS" : len(s.mn_dom_ft_10), "qp_KS" : len(s.qp_dom_ft_10), "m_slope_ft" : m_dom_slope_ft, "mn_slope_ft" : mn_dom_slope_ft, "qp_slope_ft" : qp_dom_slope_ft, "m_slope_dt" : m_dom_slope_dt, "mn_slope_dt" : mn_dom_slope_dt, "qp_slope_dt" : qp_dom_slope_dt, "m_std_error_ft" : m_dom_std_error_ft, "mn_std_error_ft" : mn_dom_std_error_ft, "qp_std_error_ft" : qp_dom_std_error_ft, "m_std_error_dt" : m_dom_std_error_dt, "mn_std_error_dt" : mn_dom_std_error_dt, "qp_std_error_dt" : qp_dom_std_error_dt, "m_intercept_ft" : m_dom_intercept_ft, "mn_intercept_ft" : mn_dom_intercept_ft, "qp_intercept_ft" : qp_dom_intercept_ft, "m_intercept_dt" : m_dom_intercept_dt, "mn_intercept_dt" : mn_dom_intercept_dt, "qp_intercept_dt" : qp_dom_intercept_dt, "qp_err" : s.qp_dom_err_10, "mn_err" : s.mn_dom_err_10, "m_err" : s.m_dom_err_10, "diagnosis" : s.diagnosis, "UPDRS" : s.UPDRS_dom, "typist" : s.typist, "qp_AT" : np.mean(s.qp_dom_dt_10), "qp_DS" : s.qp_dom_DS_10, "qp_IS" : qp_dom_IS_10, "qp_VS" : s.qp_dom_VS_10}, ignore_index=True)
                # append ndom hand        
                df = df.append({"subject_id" : s.subject_id, "hand" : s.hand, "m_se_ft" : m_ndom_se_ft, "mn_se_ft" : mn_ndom_se_ft, "qp_se_ft" : qp_ndom_se_ft, "m_se_dt" : m_ndom_se_dt, "mn_se_dt" : mn_ndom_se_dt, "qp_se_dt" : qp_ndom_se_dt, "m_KS" : len(s.m_ndom_ft_10), "mn_KS" : len(s.mn_ndom_ft_10), "qp_KS" : len(s.qp_ndom_ft_10), "m_slope_ft" : m_ndom_slope_ft, "mn_slope_ft" : mn_ndom_slope_ft, "qp_slope_ft" : qp_ndom_slope_ft, "m_slope_dt" : m_ndom_slope_dt, "mn_slope_dt" : mn_ndom_slope_dt, "qp_slope_dt" : qp_ndom_slope_dt, "m_std_error_ft" : m_ndom_std_error_ft, "mn_std_error_ft" : mn_ndom_std_error_ft, "qp_std_error_ft" : qp_ndom_std_error_dt, "m_std_error_dt" : m_ndom_std_error_dt, "mn_std_error_dt" : mn_ndom_std_error_dt, "qp_std_error_dt" : qp_ndom_std_error_dt, "m_intercept_ft" : m_ndom_intercept_ft, "mn_intercept_ft" : mn_ndom_intercept_ft, "qp_intercept_ft" : qp_ndom_intercept_ft, "m_intercept_dt" : m_ndom_intercept_dt, "mn_intercept_dt" : mn_ndom_intercept_dt, "qp_intercept_dt" : qp_ndom_intercept_dt, "qp_err" : s.qp_ndom_err_10, "mn_err" : s.mn_ndom_err_10, "m_err" : s.m_ndom_err_10, "diagnosis" : s.diagnosis, "UPDRS" : s.UPDRS_ndom, "typist" : s.typist, "qp_AT" : np.mean(s.qp_ndom_dt_10), "qp_DS" : s.qp_ndom_DS_10, "qp_IS" : qp_ndom_IS_10, "qp_VS" : s.qp_ndom_VS_10}, ignore_index=True)
        
        self.df_raw = df.copy(deep=True)
        self.df = self.preprocess(df)
        if self.selected:
            self.X = self.df[[ "vs_slope", "mn_taps", "qp_slope_ft", "m_slope_dt", "mn_slope_dt", "qp_slope_dt", "mn_std_error_ft", "qp_std_error_dt", "m_intercept_dt", "mn_intercept_dt", "qp_err", "mn_err", "m_err"]].to_numpy()
        else:
            self.X = self.df.drop(columns=["subject_id", "diagnosis", "UPDRS", "hand"]).to_numpy()
        self.y = self.df["UPDRS"].to_numpy()
        self.label_dict = dict(zip([0, 1, 2, 3], ["UPDRS_0", "UPDRS_1", "UPDRS_2","UPDRS_3"]))
        
    def loadGroup(self, time = "ft"):
        """
            Load all group features from all subjects for classification (used for experiment including assymmetry)
        """
        if (time == "ft"):
            df = pd.DataFrame(columns=["subject_id", "hand", "m_dom_vs", "mn_dom_vs", "qp_dom_vs", "m_ndom_vs", "mn_ndom_vs", "qp_ndom_vs", "m_asymmetry_slope_ft", "m_asymmetry_intercept_ft", "m_asymmetry_std_error_ft", "mn_asymmetry_slope_ft", "mn_asymmetry_intercept_ft", "mn_asymmetry_std_error_ft", "qp_asymmetry_slope_ft", "qp_asymmetry_intercept_ft", "qp_asymmetry_std_error_ft", "m_dom_se_ft", "m_ndom_se_ft", "mn_dom_se_ft", "mn_ndom_se_ft", "qp_dom_se_ft", "qp_ndom_se_ft", "m_dom_KS", "m_ndom_KS", "mn_dom_KS", "mn_ndom_KS", "qp_dom_KS", "qp_ndom_KS", "m_dom_slope_ft", "m_ndom_slope_ft", "mn_dom_slope_ft", "mn_ndom_slope_ft", "qp_dom_slope_ft", "qp_ndom_slope_ft", "m_dom_std_error_ft", "m_ndom_std_error_ft", "mn_dom_std_error_ft", "mn_ndom_std_error_ft", "qp_dom_std_error_ft", "qp_ndom_std_error_ft", "m_dom_intercept_ft", "m_ndom_intercept_ft", "mn_dom_intercept_ft", "mn_ndom_intercept_ft", "qp_dom_intercept_ft", "qp_ndom_intercept_ft", "qp_dom_err", "qp_ndom_err", "mn_dom_err", "mn_ndom_err", "m_dom_err", "m_ndom_err", "diagnosis", "UPDRS_dom", "UPDRS_ndom", "side", "typist", "qp_dom_AT_30", "qp_dom_DS_30", "qp_dom_KS_30", "qp_dom_IS_30", "qp_ndom_AT_30", "qp_ndom_DS_30", "qp_ndom_KS_30", "qp_ndom_IS_30", "qp_dom_VS_30", "qp_ndom_VS_30", "qp_dom_AT_60", "qp_dom_DS_60", "qp_dom_IS_60", "qp_ndom_AT_60", "qp_ndom_DS_60", "qp_ndom_IS_60", "qp_dom_VS_60", "qp_ndom_VS_60", "qp_dom_hesitations_ft", "qp_ndom_hesitations_ft", "mn_dom_hesitations_ft", "mn_ndom_hesitations_ft", "m_dom_hesitations_ft", "m_ndom_hesitations_ft"])
        elif (time == "dt"):
            df = pd.DataFrame(columns=["subject_id", "hand", "m_dom_vs", "mn_dom_vs", "qp_dom_vs", "m_ndom_vs", "mn_ndom_vs", "qp_ndom_vs", "m_asymmetry_slope_dt", "m_asymmetry_intercept_dt", "m_asymmetry_std_error_dt", "mn_asymmetry_slope_dt", "mn_asymmetry_intercept_dt", "mn_asymmetry_std_error_dt", "qp_asymmetry_slope_dt", "qp_asymmetry_intercept_dt", "qp_asymmetry_std_error_dt", "m_dom_se_dt", "m_ndom_se_dt", "mn_dom_se_dt", "mn_ndom_se_dt", "qp_dom_se_dt", "qp_ndom_se_dt", "m_dom_KS", "m_ndom_KS", "mn_dom_KS", "mn_ndom_KS", "qp_dom_KS", "qp_ndom_KS", "m_dom_slope_dt", "m_ndom_slope_dt", "mn_dom_slope_dt", "mn_ndom_slope_dt", "qp_dom_slope_dt", "qp_ndom_slope_dt", "m_dom_std_error_dt", "m_ndom_std_error_dt", "mn_dom_std_error_dt", "mn_ndom_std_error_dt", "qp_dom_std_error_dt", "qp_ndom_std_error_dt", "m_dom_intercept_dt", "m_ndom_intercept_dt", "mn_dom_intercept_dt", "mn_ndom_intercept_dt", "qp_dom_intercept_dt", "qp_ndom_intercept_dt", "qp_dom_err", "qp_ndom_err", "mn_dom_err", "mn_ndom_err", "m_dom_err", "m_ndom_err", "diagnosis", "UPDRS_dom", "UPDRS_ndom", "side", "typist", "qp_dom_AT_30", "qp_dom_DS_30", "qp_dom_KS_30", "qp_dom_IS_30", "qp_ndom_AT_30", "qp_ndom_DS_30", "qp_ndom_KS_30", "qp_ndom_IS_30", "qp_dom_VS_30", "qp_ndom_VS_30", "qp_dom_AT_60", "qp_dom_DS_60", "qp_dom_IS_60", "qp_ndom_AT_60", "qp_ndom_DS_60", "qp_ndom_IS_60", "qp_dom_VS_60", "qp_ndom_VS_60", "qp_dom_hesitations_dt", "qp_ndom_hesitations_dt", "mn_dom_hesitations_dt", "mn_ndom_hesitations_dt", "m_dom_hesitations_dt", "m_ndom_hesitations_dt"])

        for s in self.subjects:

            # M
            m_dom_slope_ft, m_dom_intercept_ft, _, _, m_dom_std_error_ft = stats.linregress(range(len(s.m_dom_ft)), s.m_dom_ft)
            m_dom_se_ft = getSequenceEffectScore(s.m_dom_ft)
            m_ndom_slope_ft, m_ndom_intercept_ft, _, _, m_ndom_std_error_ft = stats.linregress(range(len(s.m_ndom_ft)), s.m_ndom_ft)
            m_ndom_se_ft = getSequenceEffectScore(s.m_ndom_ft)

            m_dom_slope_dt, m_dom_intercept_dt, _, _, m_dom_std_error_dt = stats.linregress(range(len(s.m_dom_dt)), s.m_dom_dt)
            m_dom_se_dt = getSequenceEffectScore(s.m_dom_dt)
            m_ndom_slope_dt, m_ndom_intercept_dt, _, _, m_ndom_std_error_dt = stats.linregress(range(len(s.m_ndom_dt)), s.m_ndom_dt)
            m_ndom_se_dt = getSequenceEffectScore(s.m_ndom_dt)

            m_asymmetry_slope_ft = np.abs(m_dom_slope_ft - m_ndom_slope_ft)
            m_asymmetry_intercept_ft = np.abs(m_dom_intercept_ft - m_ndom_intercept_ft)
            m_asymmetry_std_error_ft = np.abs(m_dom_std_error_ft - m_ndom_std_error_ft)
            m_asymmetry_slope_dt = np.abs(m_dom_slope_dt - m_ndom_slope_dt)
            m_asymmetry_intercept_dt = np.abs(m_dom_intercept_dt - m_ndom_intercept_dt)
            m_asymmetry_std_error_dt = np.abs(m_dom_std_error_dt - m_ndom_std_error_dt)

            # MN
            mn_dom_slope_ft, mn_dom_intercept_ft, _, _, mn_dom_std_error_ft = stats.linregress(range(len(s.mn_dom_ft)), s.mn_dom_ft)
            mn_dom_se_ft = getSequenceEffectScore(s.mn_dom_ft)
            mn_ndom_slope_ft, mn_ndom_intercept_ft, _, _, mn_ndom_std_error_ft = stats.linregress(range(len(s.mn_ndom_ft)), s.mn_ndom_ft)
            mn_ndom_se_ft = getSequenceEffectScore(s.mn_ndom_ft)

            mn_dom_slope_dt, mn_dom_intercept_dt, _, _, mn_dom_std_error_dt = stats.linregress(range(len(s.mn_dom_dt)), s.mn_dom_dt)
            mn_dom_se_dt = getSequenceEffectScore(s.mn_dom_dt)
            mn_ndom_slope_dt, mn_ndom_intercept_dt, _, _, mn_ndom_std_error_dt = stats.linregress(range(len(s.mn_ndom_dt)), s.mn_ndom_dt)
            mn_ndom_se_dt = getSequenceEffectScore(s.mn_ndom_dt)

            mn_asymmetry_slope_ft = np.abs(mn_dom_slope_ft - mn_ndom_slope_ft)
            mn_asymmetry_intercept_ft = np.abs(mn_dom_intercept_ft - mn_ndom_intercept_ft)
            mn_asymmetry_std_error_ft = np.abs(mn_dom_std_error_ft - mn_ndom_std_error_ft)
            mn_asymmetry_slope_dt = np.abs(mn_dom_slope_dt - mn_ndom_slope_dt)
            mn_asymmetry_intercept_dt = np.abs(mn_dom_intercept_dt - mn_ndom_intercept_dt)
            mn_asymmetry_std_error_dt = np.abs(mn_dom_std_error_dt - mn_ndom_std_error_dt)


            # QP
            qp_dom_slope_ft, qp_dom_intercept_ft, _, _, qp_dom_std_error_ft = stats.linregress(range(len(s.qp_dom_ft)), s.qp_dom_ft)
            qp_dom_se_ft = getSequenceEffectScore(s.qp_dom_ft)
            qp_ndom_slope_ft, qp_ndom_intercept_ft, _, _, qp_ndom_std_error_ft = stats.linregress(range(len(s.qp_ndom_ft)), s.qp_ndom_ft)
            qp_ndom_se_ft = getSequenceEffectScore(s.qp_ndom_ft)

            qp_dom_slope_dt, qp_dom_intercept_dt, _, _, qp_dom_std_error_dt = stats.linregress(range(len(s.qp_dom_dt)), s.qp_dom_dt)
            qp_dom_se_dt = getSequenceEffectScore(s.qp_dom_dt)
            qp_ndom_slope_dt, qp_ndom_intercept_dt, _, _, qp_ndom_std_error_dt = stats.linregress(range(len(s.qp_ndom_dt)), s.qp_ndom_dt)
            qp_ndom_se_dt = getSequenceEffectScore(s.qp_ndom_dt)

            qp_asymmetry_slope_ft = np.abs(qp_dom_slope_ft - qp_ndom_slope_ft)
            qp_asymmetry_intercept_ft = np.abs(qp_dom_intercept_ft - qp_ndom_intercept_ft)
            qp_asymmetry_std_error_ft = np.abs(qp_dom_std_error_ft - qp_ndom_std_error_ft)
            qp_asymmetry_slope_dt = np.abs(qp_dom_slope_dt - qp_ndom_slope_dt)
            qp_asymmetry_intercept_dt = np.abs(qp_dom_intercept_dt - qp_ndom_intercept_dt)
            qp_asymmetry_std_error_dt = np.abs(qp_dom_std_error_dt - qp_ndom_std_error_dt)
            
            # Hasan 2019 Features
            qp_dom_IS_30 = statistics.variance(s.qp_dom_ft_30)
            qp_ndom_IS_30 = statistics.variance(s.qp_ndom_ft_30)
            qp_dom_IS_60 = statistics.variance(s.qp_dom_ft)
            qp_ndom_IS_60 = statistics.variance(s.qp_ndom_ft)
            
            if (time == "ft"):
                df = df.append({"subject_id" : s.subject_id, "hand" : s.hand, "m_dom_vs" : s.m_dom_vs, "mn_dom_vs" : s.mn_dom_vs, "qp_dom_vs" : s.qp_dom_vs, "m_ndom_vs" : s.m_ndom_vs, "mn_ndom_vs" : s.mn_ndom_vs, "qp_ndom_vs" : s.qp_ndom_vs, "m_asymmetry_slope_ft" : m_asymmetry_slope_ft, "m_asymmetry_intercept_ft" : m_asymmetry_intercept_ft, "m_asymmetry_std_error_ft" : m_asymmetry_std_error_ft, "mn_asymmetry_slope_ft" : mn_asymmetry_slope_ft, "mn_asymmetry_intercept_ft" : mn_asymmetry_intercept_ft, "mn_asymmetry_std_error_ft" : mn_asymmetry_std_error_ft, "qp_asymmetry_slope_ft" : qp_asymmetry_slope_ft, "qp_asymmetry_intercept_ft" : qp_asymmetry_intercept_ft, "qp_asymmetry_std_error_ft" : qp_asymmetry_std_error_ft, "m_dom_se_ft" : m_dom_se_ft, "m_ndom_se_ft" : m_ndom_se_ft, "mn_dom_se_ft" : mn_dom_se_ft, "mn_ndom_se_ft" : mn_ndom_se_ft, "qp_dom_se_ft" : qp_dom_se_ft, "qp_ndom_se_ft" : qp_ndom_se_ft, "m_dom_KS" : len(s.m_dom_ft), "m_ndom_KS" : len(s.m_ndom_ft), "mn_dom_KS" : len(s.mn_dom_ft), "mn_ndom_KS" : len(s.mn_ndom_ft), "qp_dom_KS" : len(s.qp_dom_ft), "qp_ndom_KS" : len(s.qp_ndom_ft), "m_dom_slope_ft" : m_dom_slope_ft, "m_ndom_slope_ft" : m_ndom_slope_ft, "mn_dom_slope_ft" : mn_dom_slope_ft, "mn_ndom_slope_ft" : mn_ndom_slope_ft, "qp_dom_slope_ft" : qp_dom_slope_ft, "qp_ndom_slope_ft": qp_ndom_slope_ft, "m_dom_std_error_ft" : m_dom_std_error_ft, "m_ndom_std_error_ft" : m_ndom_std_error_ft, "mn_dom_std_error_ft" : mn_dom_std_error_ft, "mn_ndom_std_error_ft" : mn_ndom_std_error_ft, "qp_dom_std_error_ft" : qp_dom_std_error_ft, "qp_ndom_std_error_ft" : qp_ndom_std_error_ft, "m_dom_intercept_ft" : m_dom_intercept_ft, "m_ndom_intercept_ft" : m_ndom_intercept_ft, "mn_dom_intercept_ft" : mn_dom_intercept_ft, "mn_ndom_intercept_ft" : mn_ndom_intercept_ft, "qp_dom_intercept_ft" : qp_dom_intercept_ft, "qp_ndom_intercept_ft" : qp_ndom_intercept_ft, "qp_dom_err" : s.qp_dom_err, "qp_ndom_err" : s.qp_ndom_err, "mn_dom_err" : s.mn_dom_err, "mn_ndom_err" : s.mn_ndom_err, "m_dom_err" : s.m_dom_err, "m_ndom_err" : s.m_ndom_err, "diagnosis" : s.diagnosis, "UPDRS_dom" : s.UPDRS_dom, "UPDRS_ndom" : s.UPDRS_ndom, "side" : s.side, "typist" : s.typist, "qp_dom_AT_30" : np.mean(s.qp_dom_dt_30), "qp_dom_DS_30" : s.qp_dom_DS_30, "qp_dom_KS_30" : len(s.qp_dom_ft_30), "qp_dom_IS_30" : qp_dom_IS_30, "qp_ndom_AT_30" : np.mean(s.qp_ndom_dt_30), "qp_ndom_DS_30" : s.qp_ndom_DS_30, "qp_ndom_KS_30" : len(s.qp_ndom_ft_30), "qp_ndom_IS_30" : qp_ndom_IS_30, "qp_dom_VS_30" : s.qp_dom_VS_30, "qp_ndom_VS_30" : s.qp_ndom_VS_30, "qp_dom_AT_60" : np.mean(s.qp_dom_dt), "qp_dom_DS_60" : s.qp_dom_DS_60, "qp_dom_IS_60" : qp_dom_IS_60, "qp_ndom_AT_60" : np.mean(s.qp_ndom_dt), "qp_ndom_DS_60" : s.qp_ndom_DS_60, "qp_ndom_IS_60" : qp_ndom_IS_60, "qp_dom_VS_60" : s.qp_dom_VS_60, "qp_ndom_VS_60" : s.qp_ndom_VS_60, "qp_dom_hesitations_ft" : s.qp_dom_out_ft, "qp_ndom_hesitations_ft" : s.qp_ndom_out_ft, "mn_dom_hesitations_ft" : s.mn_dom_out_ft, "mn_ndom_hesitations_ft" : s.mn_ndom_out_ft, "m_dom_hesitations_ft" : s.m_dom_out_ft, "m_ndom_hesitations_ft" : s.m_ndom_out_ft}, ignore_index=True)
            elif (time == "dt"):
                df = df.append({"subject_id" : s.subject_id, "hand" : s.hand, "m_dom_vs" : s.m_dom_vs, "mn_dom_vs" : s.mn_dom_vs, "qp_dom_vs" : s.qp_dom_vs, "m_ndom_vs" : s.m_ndom_vs, "mn_ndom_vs" : s.mn_ndom_vs, "qp_ndom_vs" : s.qp_ndom_vs, "m_asymmetry_slope_dt" : m_asymmetry_slope_dt, "m_asymmetry_intercept_dt" : m_asymmetry_intercept_dt, "m_asymmetry_std_error_dt" : m_asymmetry_std_error_dt, "mn_asymmetry_slope_dt" : mn_asymmetry_slope_dt, "mn_asymmetry_intercept_dt" : mn_asymmetry_intercept_dt, "mn_asymmetry_std_error_dt" : mn_asymmetry_std_error_dt, "qp_asymmetry_slope_dt" : qp_asymmetry_slope_dt, "qp_asymmetry_intercept_dt" : qp_asymmetry_intercept_dt, "qp_asymmetry_std_error_dt" : qp_asymmetry_std_error_dt, "m_dom_se_dt" : m_dom_se_dt, "m_ndom_se_dt" : m_ndom_se_dt, "mn_dom_se_dt" : mn_dom_se_dt, "mn_ndom_se_dt" : mn_ndom_se_dt, "qp_dom_se_dt" : qp_dom_se_dt, "qp_ndom_se_dt" : qp_ndom_se_dt, "m_dom_KS" : len(s.m_dom_ft), "m_ndom_KS" : len(s.m_ndom_ft), "mn_dom_KS" : len(s.mn_dom_ft), "mn_ndom_KS" : len(s.mn_ndom_ft), "qp_dom_KS" : len(s.qp_dom_ft), "qp_ndom_KS" : len(s.qp_ndom_ft), "m_dom_slope_dt" : m_dom_slope_dt, "m_ndom_slope_dt" : m_ndom_slope_dt, "mn_dom_slope_dt" : mn_dom_slope_dt, "mn_ndom_slope_dt" : mn_ndom_slope_dt, "qp_dom_slope_dt" : qp_dom_slope_dt, "qp_ndom_slope_dt": qp_ndom_slope_dt, "m_dom_std_error_dt" : m_dom_std_error_dt, "m_ndom_std_error_dt" : m_ndom_std_error_dt, "mn_dom_std_error_dt" : mn_dom_std_error_dt, "mn_ndom_std_error_dt" : mn_ndom_std_error_dt, "qp_dom_std_error_dt" : qp_dom_std_error_dt, "qp_ndom_std_error_dt" : qp_ndom_std_error_dt, "m_dom_intercept_dt" : m_dom_intercept_dt, "m_ndom_intercept_dt" : m_ndom_intercept_dt, "mn_dom_intercept_dt" : mn_dom_intercept_dt, "mn_ndom_intercept_dt" : mn_ndom_intercept_dt, "qp_dom_intercept_dt" : qp_dom_intercept_dt, "qp_ndom_intercept_dt" : qp_ndom_intercept_dt, "qp_dom_err" : s.qp_dom_err, "qp_ndom_err" : s.qp_ndom_err, "mn_dom_err" : s.mn_dom_err, "mn_ndom_err" : s.mn_ndom_err, "m_dom_err" : s.m_dom_err, "m_ndom_err" : s.m_ndom_err, "diagnosis" : s.diagnosis, "UPDRS_dom" : s.UPDRS_dom, "UPDRS_ndom" : s.UPDRS_ndom, "side" : s.side, "typist" : s.typist, "qp_dom_AT_30" : np.mean(s.qp_dom_dt_30), "qp_dom_DS_30" : s.qp_dom_DS_30, "qp_dom_KS_30" : len(s.qp_dom_ft_30), "qp_dom_IS_30" : qp_dom_IS_30, "qp_ndom_AT_30" : np.mean(s.qp_ndom_dt_30), "qp_ndom_DS_30" : s.qp_ndom_DS_30, "qp_ndom_KS_30" : len(s.qp_ndom_ft_30), "qp_ndom_IS_30" : qp_ndom_IS_30, "qp_dom_VS_30" : s.qp_dom_VS_30, "qp_ndom_VS_30" : s.qp_ndom_VS_30, "qp_dom_AT_60" : np.mean(s.qp_dom_dt), "qp_dom_DS_60" : s.qp_dom_DS_60, "qp_dom_IS_60" : qp_dom_IS_60, "qp_ndom_AT_60" : np.mean(s.qp_ndom_dt), "qp_ndom_DS_60" : s.qp_ndom_DS_60, "qp_ndom_IS_60" : qp_ndom_IS_60, "qp_dom_VS_60" : s.qp_dom_VS_60, "qp_ndom_VS_60" : s.qp_ndom_VS_60, "qp_dom_hesitations_dt" : s.qp_dom_out_dt, "qp_ndom_hesitations_dt" : s.qp_ndom_out_dt, "mn_dom_hesitations_dt" : s.mn_dom_out_dt, "mn_ndom_hesitations_dt" : s.mn_ndom_out_dt, "m_dom_hesitations_dt" : s.m_dom_out_dt, "m_ndom_hesitations_dt" : s.m_ndom_out_dt}, ignore_index=True)
                
        self.df_raw = df.copy(deep=True)
        self.df = self.preprocess(df)
        if not self.brain:
            self.df = self.df.drop(columns=["qp_dom_AT_30", "qp_dom_DS_30", "qp_dom_KS_30", "qp_dom_IS_30", "qp_ndom_AT_30", "qp_ndom_DS_30", "qp_ndom_KS_30", "qp_ndom_IS_30", "qp_dom_VS_30", "qp_ndom_VS_30"])
        if self.threeGroups:
            self.df = self.df[self.df["diagnosis"] != 1]
        if self.selected:
            self.X = self.df[["vs_slope_dom", "vs_slope_ndom", "m_dom_vs", "mn_dom_vs", "qp_dom_vs", "m_ndom_vs", "mn_ndom_vs", "qp_ndom_vs", "m_dom_se_ft", "m_ndom_se_ft", "mn_dom_se_ft", "mn_ndom_se_ft", "qp_dom_se_ft", "qp_ndom_se_ft"]].to_numpy()
        else:
            self.X = self.df.drop(columns=["subject_id", "UPDRS_dom", "UPDRS_ndom", "side", "diagnosis", "hand"]).to_numpy()
        self.y = self.df["diagnosis"].to_numpy().astype("int")
        self.label_dict = dict(zip([0, 1, 2, 3], ["PD_OFF", "PD_ON", "CA","HC"]))
        
    def split(self, random_state=0, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        return self.X_train, self.X_test, self.y_train, self.y_test
            
    def gridsearch(self, param_grid, clf, scoring=None, k=5, n_jobs=4, verbose=2, obj = "BA"):
        """
            Perform GridSearch

            Parameters:
                param_grid(dict): dictionary of possible params
                clf(classifier): classifier
                scoring(string/scorer): scoring metric
                k(int): folds of cross validation
                n_jobs(int): number of jobs to run in parallel
                verbose(int): control of verbosity
                obj(string): scorer that is used to refit data

            Output:
                GridSearchCV: GridSearchCV with best params
        """
        cv_clf = GridSearchCV(clf, param_grid, scoring=scoring, refit=obj, n_jobs=n_jobs, cv=k, verbose=verbose)
        cv_clf.fit(self.X_train, self.y_train)
        return cv_clf


    # def get_params(self, key):
    #     return self.params[key]
    #
    # def set_params(self, key, params):
    #     self.params[key] = params