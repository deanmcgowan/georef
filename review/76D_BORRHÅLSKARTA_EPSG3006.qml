<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <pipe>
    <rasterrenderer type="singlebandgray" opacity="0.8" alphaBand="-1" grayBand="1" gradient="BlackToWhite">
      <contrastEnhancement>
        <minValue>0</minValue>
        <maxValue>255</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </contrastEnhancement>
    </rasterrenderer>
    <rasterTransparency>
      <singleValuePixelList>
        <pixelListEntry min="245" max="255" percentTransparent="100" />
      </singleValuePixelList>
    </rasterTransparency>
    <brightnesscontrast brightness="0" contrast="0" gamma="1" />
    <huesaturation colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeOn="0" colorizeStrength="100" grayscaleMode="0" saturation="0" />
    <rasterresampler maxOversampling="2" />
  </pipe>
  <layerTransparency>20</layerTransparency>
  <blendMode>0</blendMode>
  <layerGeometryType>4</layerGeometryType>
</qgis>
