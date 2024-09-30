import pandas as pd
import geopandas as gpd

# code to create the final.csv (only need to do once)
gdf = gpd.read_file('/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/data_with_slopes.gpkg')
df = pd.read_csv('/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/data_with_era5.csv')
df['storm'] = df['storm'].str.capitalize()

final_df = gdf.merge(df, left_on=['center_centerLat', 'center_centerLon', 'stormName'], right_on=['lat', 'lon', 'storm'], how='left')
final_df = final_df.dropna()
final_df.to_csv('/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/final.csv', index=False)
