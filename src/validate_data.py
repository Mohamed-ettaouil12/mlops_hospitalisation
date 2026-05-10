import great_expectations as gx
import pandas as pd
import os

DATA_DIR   = '/home/tawil/Bureau/pfa/data set/sample 1/'
SAMPLE_NUM = 1

print("📂 Chargement des données pour validation...")
ben08 = pd.read_csv(
    os.path.join(DATA_DIR, f'DE1_0_2008_Beneficiary_Summary_File_Sample_{SAMPLE_NUM}.csv'),
    low_memory=False
)

context = gx.get_context()
datasource = context.sources.add_pandas("cms_datasource")
asset = datasource.add_dataframe_asset("beneficiary_2008")
batch_request = asset.build_batch_request(dataframe=ben08)

context.add_or_update_expectation_suite("cms_raw_suite")

validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="cms_raw_suite"
)

validator.expect_column_values_to_be_unique("DESYNPUF_ID")
validator.expect_column_values_to_be_in_set("BENE_SEX_IDENT_CD", value_set=[1, 2])

validator.save_expectation_suite(discard_failed_expectations=False)
results = validator.validate()

print(f"✅ Tests réussis  : {results['statistics']['successful_expectations']}")
print(f"❌ Tests échoués  : {results['statistics']['unsuccessful_expectations']}")
print(f"🎯 Succès global  : {results['success']}")
