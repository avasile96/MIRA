t = Tiff('copd4_iBHCT.tif','w');

setTag(t,'Photometric',Tiff.Photometric.MinIsBlack);
setTag(t,'Compression',Tiff.Compression.None);
setTag(t,'ResolutionUnit',Tiff.ResolutionUnit.Centimeter);
setTag(t,'XResolution',0.00625);
setTag(t,'YResolution',0.00625);
setTag(t,'BitsPerSample',16);
setTag(t,'SamplesPerPixel',126); % Change for number of slices
setTag(t,'SampleFormat',Tiff.SampleFormat.Int);
setTag(t,'ExtraSamples',Tiff.ExtraSamples.Unspecified);
setTag(t,'ImageLength',512); % Change for image size
setTag(t,'ImageWidth',512); % Change for image size
setTag(t,'TileLength',32);
setTag(t,'TileWidth',32);
setTag(t,'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);

write(t,copd4_iBHCT);
close(t);


