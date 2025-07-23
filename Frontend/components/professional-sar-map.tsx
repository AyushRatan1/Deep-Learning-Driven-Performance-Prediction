"use client"

import React, { useEffect, useRef, useState, useCallback } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { MapPin, Navigation, Target, Zap, Shield, AlertTriangle, Activity, Satellite } from "lucide-react";
import { useSAR } from "@/components/sar-context";

// Mapbox access token
mapboxgl.accessToken = 'pk.eyJ1IjoiYXl1c2hyYXRhbiIsImEiOiJjbWJsdmtxcnMxNzByMnJvaDM4dXNra3VsIn0.awICGO_cB5OUVHzK0ea2AQ';

interface AntennaLocation {
  latitude: number;
  longitude: number;
  altitude: number;
  name: string;
  power: number; // in mW
  frequency: number; // in GHz
}

interface SARZone {
  center: [number, number];
  radius: number;
  sarLevel: number;
  safetyStatus: 'safe' | 'caution' | 'warning' | 'danger';
}

interface ProfessionalSARMapProps {
  className?: string;
  antennaLocation?: AntennaLocation;
  onLocationChange?: (location: AntennaLocation) => void;
}

export function ProfessionalSARMap({ 
  className, 
  antennaLocation,
  onLocationChange 
}: ProfessionalSARMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const { state } = useSAR();
  
  // Map state
  const [isMapLoaded, setIsMapLoaded] = useState(false);
  const [currentLocation, setCurrentLocation] = useState<AntennaLocation>({
    latitude: 37.7749, // San Francisco default
    longitude: -122.4194,
    altitude: 10,
    name: "Antenna Site",
    power: 100, // mW
    frequency: 2.45 // GHz
  });
  
  // UI state
  const [mapStyle, setMapStyle] = useState('mapbox://styles/mapbox/satellite-v9');
  const [showSafetyZones, setShowSafetyZones] = useState(true);
  const [showRadiationPattern, setShowRadiationPattern] = useState(true);
  const [showTerrainAnalysis, setShowTerrainAnalysis] = useState(false);
  const [sarThreshold, setSarThreshold] = useState([1.6]); // FCC limit
  const [isLocationDetecting, setIsLocationDetecting] = useState(false);
  
  // SAR calculation state
  const [sarZones, setSarZones] = useState<SARZone[]>([]);
  const [maxSarDistance, setMaxSarDistance] = useState(100); // meters
  const [sarResolution, setSarResolution] = useState(50); // calculation points
  
  // Coordinates input
  const [manualCoords, setManualCoords] = useState({
    lat: currentLocation.latitude.toString(),
    lng: currentLocation.longitude.toString(),
    alt: currentLocation.altitude.toString()
  });

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: mapStyle,
        center: [currentLocation.longitude, currentLocation.latitude],
        zoom: 16,
        pitch: 45,
        bearing: 0,
        antialias: true,
        preserveDrawingBuffer: true
      });

      map.current.on('load', () => {
        try {
          setIsMapLoaded(true);
          
          // Add 3D buildings safely
          const checkAndAddBuildings = () => {
            if (map.current?.getLayer && map.current.getLayer('building')) {
              map.current.setLayoutProperty('building', 'visibility', 'visible');
              map.current.setPaintProperty('building', 'fill-extrusion-opacity', 0.8);
            }
          };
          
          // Delay building setup to ensure style is fully loaded
          setTimeout(checkAndAddBuildings, 1000);
          
          // Add terrain and sky safely
          const addTerrainFeatures = () => {
            try {
              if (map.current && map.current.isStyleLoaded()) {
                map.current.addSource('mapbox-dem', {
                  type: 'raster-dem',
                  url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
                  tileSize: 512,
                  maxzoom: 14
                });
                
                map.current.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 });
                
                map.current.addLayer({
                  id: 'sky',
                  type: 'sky',
                  paint: {
                    'sky-type': 'atmosphere',
                    'sky-atmosphere-sun-intensity': 5
                  }
                });
              }
            } catch (error) {
              console.warn('Terrain features not available:', error);
            }
          };
          
          // Delay terrain setup
          setTimeout(addTerrainFeatures, 2000);
          
          // Initialize SAR overlay
          setTimeout(() => {
            if (map.current && map.current.isStyleLoaded()) {
              initializeSAROverlay();
            }
          }, 1500);
          
        } catch (error) {
          console.error('Error in map load handler:', error);
        }
      });

      map.current.on('error', (e) => {
        console.error('Mapbox error:', e);
      });

      // Add navigation controls
      map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');
      map.current.addControl(new mapboxgl.FullscreenControl(), 'top-right');
      map.current.addControl(new mapboxgl.ScaleControl(), 'bottom-left');

      // Add click handler for antenna placement
      map.current.on('click', (e) => {
        if (e.lngLat) {
          const { lng, lat } = e.lngLat;
          updateAntennaLocation({
            ...currentLocation,
            latitude: lat,
            longitude: lng
          });
        }
      });

    } catch (error) {
      console.error('Error initializing map:', error);
    }

    return () => {
      try {
        if (map.current) {
          map.current.remove();
          map.current = null;
          setIsMapLoaded(false);
        }
      } catch (error) {
        console.error('Error cleaning up map:', error);
      }
    };
  }, []);

  // Update map style
  useEffect(() => {
    if (map.current && isMapLoaded) {
      try {
        map.current.setStyle(mapStyle);
        map.current.once('styledata', () => {
          setTimeout(() => {
            if (map.current && map.current.isStyleLoaded()) {
              // Re-initialize overlay after style change
              setIsMapLoaded(true);
            }
          }, 500);
        });
      } catch (error) {
        console.error('Error updating map style:', error);
      }
    }
  }, [mapStyle, isMapLoaded]);

  // Update antenna location
  const updateAntennaLocation = useCallback((location: AntennaLocation) => {
    setCurrentLocation(location);
    setManualCoords({
      lat: location.latitude.toString(),
      lng: location.longitude.toString(),
      alt: location.altitude.toString()
    });
    
    if (map.current) {
      map.current.flyTo({
        center: [location.longitude, location.latitude],
        zoom: 16,
        duration: 1000
      });
    }
    
    onLocationChange?.(location);
    calculateSARZones(location);
  }, [onLocationChange]);

  // Get device location
  const getDeviceLocation = useCallback(() => {
    setIsLocationDetecting(true);
    
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by this browser');
      setIsLocationDetecting(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const location: AntennaLocation = {
          ...currentLocation,
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          altitude: position.coords.altitude || 10,
          name: "Current Device Location"
        };
        updateAntennaLocation(location);
        setIsLocationDetecting(false);
      },
      (error) => {
        console.error('Error getting location:', error);
        alert('Could not get current location. Please check permissions.');
        setIsLocationDetecting(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 60000
      }
    );
  }, [currentLocation, updateAntennaLocation]);

  // Apply manual coordinates
  const applyManualCoordinates = useCallback(() => {
    const lat = parseFloat(manualCoords.lat);
    const lng = parseFloat(manualCoords.lng);
    const alt = parseFloat(manualCoords.alt);
    
    if (isNaN(lat) || isNaN(lng) || isNaN(alt)) {
      alert('Please enter valid coordinates');
      return;
    }
    
    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      alert('Coordinates out of range');
      return;
    }
    
    updateAntennaLocation({
      ...currentLocation,
      latitude: lat,
      longitude: lng,
      altitude: alt,
      name: "Manual Location"
    });
  }, [manualCoords, currentLocation, updateAntennaLocation]);

  // Calculate SAR zones around antenna using backend API
  const calculateSARZones = useCallback(async (location: AntennaLocation) => {
    try {
      // Get current antenna parameters from state or use defaults
      const parameters = state.parameters || {
        substrate_thickness: 1.6,
        substrate_permittivity: 4.4,
        patch_width: 35.0,
        patch_length: 40.0,
        bending_radius: 50.0,
        power_density: 1.0
      };

      // Prepare API request
      const requestData = {
        location: {
          name: location.name,
          geolocation: {
            latitude: location.latitude,
            longitude: location.longitude,
            altitude: location.altitude,
            accuracy: 5.0
          },
          power: location.power,
          frequency: location.frequency
        },
        parameters: parameters,
        analysis_range: maxSarDistance,
        resolution: sarResolution,
        include_terrain: showTerrainAnalysis
      };

      // Check if backend is available first (with shorter timeout)
      try {
        const healthCheck = await fetch('http://localhost:8000/health', { 
          method: 'GET',
          mode: 'cors',
          signal: AbortSignal.timeout(2000)
        });
        
        if (!healthCheck.ok) {
          console.warn('Backend health check failed, using fallback calculation');
          throw new Error('Backend API not responding');
        }
      } catch (healthError) {
        console.warn('Backend not reachable, using local calculation:', healthError);
        throw new Error('Backend API not available');
      }

      // Call backend API with error handling
      const response = await fetch('http://localhost:8000/api/sar-map', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestData),
        mode: 'cors',
        credentials: 'same-origin'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API call failed: ${response.status} - ${errorText}`);
      }

      const sarMapData = await response.json();
      
      // Convert backend response to our zone format
      const zones: SARZone[] = sarMapData.zones.map((zone: any) => ({
        center: [zone.center_lng, zone.center_lat],
        radius: zone.radius,
        sarLevel: zone.sar_level,
        safetyStatus: zone.safety_status
      }));
      
      setSarZones(zones);
      updateSAROverlay(zones);
      
      console.log('SAR Analysis Results:', {
        maxSAR: sarMapData.max_sar,
        safeDistance: sarMapData.safe_distance,
        compliance: sarMapData.compliance_status
      });
      
    } catch (error) {
      console.error('Error calculating SAR zones:', error);
      
      // Show user-friendly error message
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        console.warn('Backend API not available, using fallback calculation');
      }
      
      // Fallback to simplified calculation
      const zones: SARZone[] = [];
      const resolution = sarResolution;
      const maxDistance = maxSarDistance;
      
      // Use current prediction or default values
      const sarValue = state.currentPrediction?.sar_value || 0.8;
      const gain = state.currentPrediction?.gain || 4.0;
      const frequency = location.frequency;
      const power = location.power / 1000; // Convert mW to W
      
      // Calculate SAR at different distances (simplified model)
      for (let distance = 1; distance <= maxDistance; distance += Math.max(1, maxDistance / resolution)) {
        // Basic SAR calculation: SAR ∝ Power * Gain / (4π * distance²)
        const powerDensity = (power * Math.pow(10, gain/10)) / (4 * Math.PI * Math.pow(distance/100, 2));
        const tissueProperty = 0.8; // Simplified tissue interaction factor
        const calculatedSAR = powerDensity * tissueProperty * frequency;
        
        let safetyStatus: 'safe' | 'caution' | 'warning' | 'danger' = 'safe';
        if (calculatedSAR > 1.6) safetyStatus = 'danger';
        else if (calculatedSAR > 1.2) safetyStatus = 'warning';
        else if (calculatedSAR > 0.8) safetyStatus = 'caution';
        
        zones.push({
          center: [location.longitude, location.latitude],
          radius: distance,
          sarLevel: calculatedSAR,
          safetyStatus
        });
      }
      
      setSarZones(zones);
      updateSAROverlay(zones);
    }
  }, [sarResolution, maxSarDistance, showTerrainAnalysis, state.currentPrediction, state.parameters]);

  // Initialize SAR overlay on map
  const initializeSAROverlay = useCallback(() => {
    if (!map.current || !isMapLoaded) return;

    // Remove existing SAR layers in correct order (layers first, then sources)
    if (map.current.getLayer('sar-zones-stroke')) {
      map.current.removeLayer('sar-zones-stroke');
    }
    if (map.current.getLayer('sar-zones')) {
      map.current.removeLayer('sar-zones');
    }
    if (map.current.getSource('sar-zones')) {
      map.current.removeSource('sar-zones');
    }

    // Remove antenna layers
    if (map.current.getLayer('antenna-icon')) {
      map.current.removeLayer('antenna-icon');
    }
    if (map.current.getLayer('antenna-marker')) {
      map.current.removeLayer('antenna-marker');
    }
    if (map.current.getSource('antenna-marker')) {
      map.current.removeSource('antenna-marker');
    }

    // Add antenna point source and layer
    map.current.addSource('antenna-marker', {
      type: 'geojson',
      data: {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [currentLocation.longitude, currentLocation.latitude]
        },
        properties: {
          name: currentLocation.name,
          power: currentLocation.power,
          frequency: currentLocation.frequency
        }
      }
    });

    map.current.addLayer({
      id: 'antenna-marker',
      type: 'circle',
      source: 'antenna-marker',
      paint: {
        'circle-radius': 12,
        'circle-color': '#ff4444',
        'circle-stroke-color': '#ffffff',
        'circle-stroke-width': 3,
        'circle-opacity': 0.9
      }
    });

    // Add antenna icon layer
    map.current.addLayer({
      id: 'antenna-icon',
      type: 'symbol',
      source: 'antenna-marker',
      layout: {
        'icon-image': 'communications-15', // Mapbox default icon
        'icon-size': 2,
        'icon-offset': [0, 0],
        'text-field': ['get', 'name'],
        'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
        'text-offset': [0, 2],
        'text-anchor': 'top',
        'text-size': 12
      },
      paint: {
        'text-color': '#ffffff',
        'text-halo-color': '#000000',
        'text-halo-width': 1
      }
    });

  }, [isMapLoaded, currentLocation]);

  // Update SAR overlay with zones
  const updateSAROverlay = useCallback((zones: SARZone[]) => {
    if (!map.current || !isMapLoaded) return;
    
    // If safety zones are disabled, remove existing layers
    if (!showSafetyZones) {
      if (map.current.getLayer('sar-zones-stroke')) {
        map.current.removeLayer('sar-zones-stroke');
      }
      if (map.current.getLayer('sar-zones')) {
        map.current.removeLayer('sar-zones');
      }
      if (map.current.getSource('sar-zones')) {
        map.current.removeSource('sar-zones');
      }
      return;
    }

    const features = zones.map((zone, index) => {
      const color = 
        zone.safetyStatus === 'safe' ? '#10b981' :
        zone.safetyStatus === 'caution' ? '#f59e0b' :
        zone.safetyStatus === 'warning' ? '#ef4444' :
        '#dc2626';

      // Create circle polygon
      const center = turf.point(zone.center);
      const circle = turf.circle(center, zone.radius / 1000, { units: 'kilometers' });

      return {
        ...circle,
        properties: {
          ...circle.properties,
          sarLevel: zone.sarLevel,
          safetyStatus: zone.safetyStatus,
          color: color,
          opacity: zone.safetyStatus === 'safe' ? 0.1 : 
                  zone.safetyStatus === 'caution' ? 0.2 :
                  zone.safetyStatus === 'warning' ? 0.3 : 0.4
        }
      };
    });

    if (map.current.getSource('sar-zones')) {
      (map.current.getSource('sar-zones') as mapboxgl.GeoJSONSource).setData({
        type: 'FeatureCollection',
        features: features
      });
    } else {
      map.current.addSource('sar-zones', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: features
        }
      });

      map.current.addLayer({
        id: 'sar-zones',
        type: 'fill',
        source: 'sar-zones',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': ['get', 'opacity']
        }
      });

      map.current.addLayer({
        id: 'sar-zones-stroke',
        type: 'line',
        source: 'sar-zones',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 1,
          'line-opacity': 0.8
        }
      });
    }
  }, [isMapLoaded, showSafetyZones]);

  // Helper function for creating circles (simplified turf.js equivalent)
  const turf = {
    point: (coordinates: number[]) => ({
      type: 'Feature' as const,
      geometry: { type: 'Point' as const, coordinates },
      properties: {}
    }),
    circle: (center: any, radius: number, options: any = {}) => {
      const coordinates = [];
      const steps = 64;
      for (let i = 0; i < steps; i++) {
        const angle = (i / steps) * 2 * Math.PI;
        const dx = radius * Math.cos(angle);
        const dy = radius * Math.sin(angle);
        coordinates.push([
          center.geometry.coordinates[0] + dx / 111.32, // Rough conversion
          center.geometry.coordinates[1] + dy / 110.54
        ]);
      }
      coordinates.push(coordinates[0]); // Close the polygon
      
      return {
        type: 'Feature' as const,
        geometry: { type: 'Polygon' as const, coordinates: [coordinates] },
        properties: options.properties || {}
      };
    }
  };

  // Update overlays when settings change
  useEffect(() => {
    if (isMapLoaded) {
      updateSAROverlay(sarZones);
    }
  }, [showSafetyZones, showRadiationPattern, isMapLoaded, sarZones, updateSAROverlay]);

  // Calculate initial SAR zones
  useEffect(() => {
    if (isMapLoaded) {
      calculateSARZones(currentLocation);
    }
  }, [isMapLoaded, currentLocation, calculateSARZones]);

  return (
    <div className={`w-full h-full ${className}`}>
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-full">
        {/* Map Container */}
        <div className="lg:col-span-3 relative">
          <Card className="h-full">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <Satellite className="h-5 w-5" />
                Professional SAR Coverage Analysis
                <Badge variant="secondary" className="ml-auto">
                  {currentLocation.frequency} GHz • {currentLocation.power} mW
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 h-[calc(100%-80px)]">
              <div 
                ref={mapContainer} 
                className="w-full h-full rounded-lg overflow-hidden"
                style={{ minHeight: '500px' }}
              />
              
              {/* Map overlay controls */}
              <div className="absolute top-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg p-3 space-y-2">
                <div className="flex items-center gap-2 text-white text-sm">
                  <MapPin className="h-4 w-4" />
                  {currentLocation.name}
                </div>
                <div className="text-xs text-gray-300">
                  {currentLocation.latitude.toFixed(6)}, {currentLocation.longitude.toFixed(6)}
                </div>
                <div className="text-xs text-gray-300">
                  Altitude: {currentLocation.altitude}m
                </div>
              </div>

              {/* Safety legend */}
              {showSafetyZones && (
                <div className="absolute bottom-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg p-3 space-y-2">
                  <div className="text-white text-sm font-medium">SAR Safety Zones</div>
                  <div className="space-y-1 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full opacity-40"></div>
                      <span className="text-gray-300">Safe (&lt; 0.8 W/kg)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full opacity-40"></div>
                      <span className="text-gray-300">Caution (0.8-1.2 W/kg)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full opacity-40"></div>
                      <span className="text-gray-300">Warning (1.2-1.6 W/kg)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-700 rounded-full opacity-40"></div>
                      <span className="text-gray-300">Danger (&gt; 1.6 W/kg)</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Control Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Location Control</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs defaultValue="auto" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="auto">Auto</TabsTrigger>
                  <TabsTrigger value="manual">Manual</TabsTrigger>
                </TabsList>
                
                <TabsContent value="auto" className="space-y-3">
                  <Button 
                    onClick={getDeviceLocation}
                    disabled={isLocationDetecting}
                    className="w-full"
                  >
                    <Navigation className="h-4 w-4 mr-2" />
                    {isLocationDetecting ? 'Detecting...' : 'Use Device Location'}
                  </Button>
                  
                  <p className="text-xs text-muted-foreground">
                    Click anywhere on the map to place antenna
                  </p>
                </TabsContent>
                
                <TabsContent value="manual" className="space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <Label htmlFor="lat" className="text-xs">Latitude</Label>
                      <Input
                        id="lat"
                        value={manualCoords.lat}
                        onChange={(e) => setManualCoords(prev => ({ ...prev, lat: e.target.value }))}
                        placeholder="37.7749"
                        className="text-xs"
                      />
                    </div>
                    <div>
                      <Label htmlFor="lng" className="text-xs">Longitude</Label>
                      <Input
                        id="lng"
                        value={manualCoords.lng}
                        onChange={(e) => setManualCoords(prev => ({ ...prev, lng: e.target.value }))}
                        placeholder="-122.4194"
                        className="text-xs"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <Label htmlFor="alt" className="text-xs">Altitude (m)</Label>
                    <Input
                      id="alt"
                      value={manualCoords.alt}
                      onChange={(e) => setManualCoords(prev => ({ ...prev, alt: e.target.value }))}
                      placeholder="10"
                      className="text-xs"
                    />
                  </div>
                  
                  <Button onClick={applyManualCoordinates} className="w-full" size="sm">
                    <Target className="h-4 w-4 mr-2" />
                    Apply Coordinates
                  </Button>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Antenna Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-sm">Power Level: {currentLocation.power} mW</Label>
                <Slider
                  value={[currentLocation.power]}
                  onValueChange={(value) => 
                    updateAntennaLocation({ ...currentLocation, power: value[0] })
                  }
                  max={1000}
                  min={1}
                  step={1}
                  className="mt-2"
                />
              </div>
              
              <div>
                <Label className="text-sm">Frequency: {currentLocation.frequency} GHz</Label>
                <Select
                  value={currentLocation.frequency.toString()}
                  onValueChange={(value) => 
                    updateAntennaLocation({ ...currentLocation, frequency: parseFloat(value) })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0.45">0.45 GHz (UHF)</SelectItem>
                    <SelectItem value="0.9">0.9 GHz (GSM)</SelectItem>
                    <SelectItem value="2.4">2.4 GHz (ISM)</SelectItem>
                    <SelectItem value="5.8">5.8 GHz (ISM)</SelectItem>
                    <SelectItem value="28">28 GHz (5G mmWave)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Display Options</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-sm">Map Style</Label>
                <Select value={mapStyle} onValueChange={setMapStyle}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mapbox://styles/mapbox/satellite-v9">Satellite</SelectItem>
                    <SelectItem value="mapbox://styles/mapbox/streets-v12">Streets</SelectItem>
                    <SelectItem value="mapbox://styles/mapbox/outdoors-v12">Outdoors</SelectItem>
                    <SelectItem value="mapbox://styles/mapbox/dark-v11">Dark</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Safety Zones</Label>
                  <Switch 
                    checked={showSafetyZones}
                    onCheckedChange={setShowSafetyZones}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Radiation Pattern</Label>
                  <Switch 
                    checked={showRadiationPattern}
                    onCheckedChange={setShowRadiationPattern}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Terrain Analysis</Label>
                  <Switch 
                    checked={showTerrainAnalysis}
                    onCheckedChange={setShowTerrainAnalysis}
                  />
                </div>
              </div>

              <Separator />

              <div>
                <Label className="text-sm">Analysis Range: {maxSarDistance}m</Label>
                <Slider
                  value={[maxSarDistance]}
                  onValueChange={(value) => setMaxSarDistance(value[0])}
                  max={1000}
                  min={10}
                  step={10}
                  className="mt-2"
                />
              </div>

              <div>
                <Label className="text-sm">Resolution: {sarResolution} points</Label>
                <Slider
                  value={[sarResolution]}
                  onValueChange={(value) => setSarResolution(value[0])}
                  max={200}
                  min={20}
                  step={10}
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Safety Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {sarZones.length > 0 && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Max SAR Level:</span>
                      <span className="font-mono">
                        {Math.max(...sarZones.map(z => z.sarLevel)).toFixed(3)} W/kg
                      </span>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span>Safe Distance:</span>
                      <span className="font-mono">
                        {sarZones.find(z => z.safetyStatus === 'safe')?.radius.toFixed(1) || '0'} m
                      </span>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span>Warning Zones:</span>
                      <span className="font-mono">
                        {sarZones.filter(z => z.safetyStatus === 'warning' || z.safetyStatus === 'danger').length}
                      </span>
                    </div>
                  </div>
                  
                  <Progress 
                    value={Math.min(100, (Math.max(...sarZones.map(z => z.sarLevel)) / 1.6) * 100)}
                    className="h-2"
                  />
                  
                  <p className="text-xs text-muted-foreground">
                    {Math.max(...sarZones.map(z => z.sarLevel)) > 1.6 
                      ? "⚠️ Exceeds FCC safety limit" 
                      : "✅ Within FCC safety limits"}
                  </p>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 