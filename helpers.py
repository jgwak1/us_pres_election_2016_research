def substring_find_replacer(dt, colname, substring_ls, others_str, verbose = False):
    
    data = dt.copy()
    
    data[colname] = data.apply(lambda row: row[colname].lower(), axis=1)

    for substr in substring_ls:
        data[colname] = data[colname].apply(lambda x: substr if substr in x else x)
        
        if (verbose is True):
            print("\n"); print(data[colname].value_counts())
        
    data[colname] = data[colname].apply(lambda x: others_str if x not in substring_ls else x)

    if (verbose is True):
        print("\n"); print(data[colname].value_counts())
    
    return data