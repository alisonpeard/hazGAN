import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(18,8)) 
ax  = plt.axes(projection =ccrs.PlateCarree())   
ax.add_feature(cartopy.feature.LAND,facecolor='wheat') 
ax.add_feature(cartopy.feature.OCEAN) 
ax.add_feature(cartopy.feature.STATES, linestyle='-',lw=1.0,edgecolor='white') 
ax.add_feature(cartopy.feature.BORDERS, linestyle='-',lw=2.5,edgecolor='white') 
gp = ds.isel(time=0).gpm.plot.pcolormesh('lon','lat',ax=ax,
             infer_intervals=True,vmin=0,vmax=1600,cmap='jet',extend='both') 
ax.coastlines(resolution='50m',color='white') 
ax.add_feature(cartopy.feature.RIVERS) 
gl = ax.gridlines(color='gray',alpha=0.6,draw_labels=True) 
gl.xlabels_top, gl.ylabels_right = False, False
gl.xlabel_style, gl.ylabel_style = {'fontsize': 30}, {'fontsize': 40}
ax.yaxis.tick_right() 
ax.set_ylim([18,32]) 
ax.set_xlim([-100,-77])  
ax.set_title('GPM Accumulated Rain (12Z Aug24 - 12Z Aug31)', fontsize=16) 
ax.set_aspect('equal') 
plt.show()