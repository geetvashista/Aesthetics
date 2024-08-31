# notes for tamar how to select epochs after preprocessing - navigate to epoch no. through folders: annotations > descriptions (view array) > find list no. corresponding with epochs (sanity check by looking in variables after saving p0x_hig and p0x low as variables)
epochs = epochs.resample(250, npad="auto")
p02_hig = epochs[7]
p02_low = epochs[16]
p02_hig.save('p02_hig')
p02_low.save('p02_low')

#note these save in projects folder (otherwise 'p02_hig etc.' could be changed to filepath of desired output folder)

#plot stc after beamformer code
stc_hig.plot(hemi='both')
stc_low.plot(hemi='both')
