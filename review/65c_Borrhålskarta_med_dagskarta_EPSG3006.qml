<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <pipe>
    <rasterrenderer type="multibandcolor" opacity="0.8" alphaBand="-1" redBand="1" greenBand="2" blueBand="3">
      <redContrastEnhancement>
        <minValue>0</minValue>
        <maxValue>255</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </redContrastEnhancement>
      <greenContrastEnhancement>
        <minValue>0</minValue>
        <maxValue>255</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </greenContrastEnhancement>
      <blueContrastEnhancement>
        <minValue>0</minValue>
        <maxValue>255</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </blueContrastEnhancement>
    </rasterrenderer>
    <rasterTransparency>
      <singleValuePixelList>
        <pixelListEntry min="245" max="255" percentTransparent="100" band="1" />
        <pixelListEntry min="245" max="255" percentTransparent="100" band="2" />
        <pixelListEntry min="245" max="255" percentTransparent="100" band="3" />
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
