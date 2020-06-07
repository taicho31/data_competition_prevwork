import itertools
def blute_force_fe(df, variables):
    for v in itertools.combinations(variables, 2):
        df["sum"+str(v[0])+"_"+str(v[1])] = df[v[0]] + df[v[1]]
        df["diff"+str(v[0])+"_"+str(v[1])] = df[v[0]] - df[v[1]]
        df["abs_diff"+str(v[0])+"_"+str(v[1])] = np.abs(df[v[0]] - df[v[1]])
        df["mul"+str(v[0])+"_"+str(v[1])] = df[v[0]] * df[v[1]]
        df["div"+str(v[0])+"_"+str(v[1])] = df[v[0]] / df[v[1]]   
    return df

params = ["params0", "params1","params2", "params3","params4", "params5","params6"]
#new_train = blute_force_fe(new_train, params)
#new_test = blute_force_fe(new_test, params)
