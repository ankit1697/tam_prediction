# TAM Prediction

Predicting the Total Addressable Market of a brand using location-based data

## Libraries Used

 * Geopandas - Used for geometric data frame manipulations similar to Pandas
 * H3 - A geospatial indexing system, used to aggregate lat-lngs at a hex level
 * KeplerGL - Used to visualize the lat-lngs, hexes, geometries, etc. on a map

### Sample KeplerGl Map

![Alt text](https://assets-global.website-files.com/5f2a93fe880654a977c51043/60305b4c4e81d9b9e64d7334_kepler_sf_jupyter.png "Kepler")

How to use:

	pip install keplergl
	
	from keplergl import KeplerGl
	
	KeplerGl(data={'a':data})
	
##### Note: The dataframe should either have a latitude/longitude, geometry or hex column


### H3 hexes

 * Provides an index to every lat-lng
 * often represented as a 15-character (or 16-character) hexadecimal string, like '8928308280fffff'

For further reference: 	[H3](https://h3geo.org)

 
 