from pyproj import Transformer

class GNSStoUTMConverter:
    def __init__(self):
        # Initialize the converter
        pass

    def determine_utm_zone(self, lon):
        # Determine the UTM zone based on longitude
        return int((lon + 180) / 6) + 1

    def convert(self, lat, lon):
        # Convert GNSS coordinates to UTM
        zone_number = self.determine_utm_zone(lon)
        
        # Create a transformer object
        transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 GNSS coordinate system
            f"+proj=utm +zone={zone_number} +datum=WGS84",  # UTM coordinate system
            always_xy=True
        )
        
        # Perform the transformation
        easting, northing = transformer.transform(lon, lat)
        
        return easting, northing, zone_number

# Example usage
if __name__ == "__main__":
    converter = GNSStoUTMConverter()
    
    # San Francisco coordinates
    lat, lon = 37.7749, -122.4194
    
    easting, northing, zone = converter.convert(lat, lon)
    
    print(f"UTM coordinates: {easting:.2f} E, {northing:.2f} N, Zone {zone}")
