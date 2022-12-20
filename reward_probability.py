def reward_prob_arr(df):

    k=len(df.columns)
    rows=len(df)
    reward_vector=[0]*k


    count_col=0

    column_num=0
    for column_name in df.columns:
        column_current=df[column_name]
        count_col=(column_current!=0).sum()
        reward_vector[column_num]=count_col/rows
        column_num=column_num+1


    return reward_vector
