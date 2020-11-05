import pandas


def import_dataset(filename):
    col_list = ["company_profile", "description", "requirements", "fraudulent"]
    df = pandas.read_csv(filename, usecols=col_list, encoding="utf-8")
    return df


df = import_dataset('fake_job_postings.csv')
columns = ['text', 'fake']
index = range(0, len(df))
df_adapted = df_ = pandas.DataFrame(index=index, columns=columns)
for index, row in df.iterrows():
    row = row.copy()
    string = str(row["company_profile"]) + str(row["description"]) + str(row["requirements"])
    df_adapted.loc[index, "text"] = string
    df_adapted.loc[index, "fake"] = row["fraudulent"]


df_adapted.to_csv('fake_job_postings_processed.csv', index=False)
