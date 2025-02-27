let map = L.map('map').setView([28.6139, 77.2090], 12); // Default: New Delhi

// Load OpenStreetMap Tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Store blood bank markers
let bloodBankMarkers = [];
let routeLayer; // Store the route layer

// Create a container for the list of blood banks
let listContainer = document.createElement('div');
listContainer.id = 'bloodBankList';
listContainer.style.padding = '10px';
listContainer.style.maxHeight = '300px';
listContainer.style.overflowY = 'auto';
document.body.appendChild(listContainer);

// Get User Location (Real-Time Tracking)
let userMarker;
if (navigator.geolocation) {
    navigator.geolocation.watchPosition(position => {
        let userLocation = [position.coords.latitude, position.coords.longitude];

        if (!userMarker) {
            userMarker = L.marker(userLocation, {icon: L.icon({iconUrl: "http://maps.google.com/mapfiles/ms/icons/blue-dot.png"})})
                .addTo(map)
                .bindPopup("You are here")
                .openPopup();
        } else {
            userMarker.setLatLng(userLocation);
        }

        map.setView(userLocation, 14);
        fetchBloodBanks(userLocation, 10000); // Start with 10 km radius
    }, error => {
        alert("Geolocation failed. Using default location.");
        fetchBloodBanks([28.6139, 77.2090], 10000); // Default to Delhi
    });
} else {
    alert("Geolocation is not supported by your browser.");
    fetchBloodBanks([28.6139, 77.2090], 10000); // Default location
}

// Find the Fastest Route to the Selected Blood Bank and Open Google Maps for Navigation
function findRoute(lat, lng) {
    if (!userMarker) {
        alert("User location not found.");
        return;
    }

    let userLocation = userMarker.getLatLng();

    // Open Google Maps for step-by-step navigation
    let googleMapsURL = `https://www.google.com/maps/dir/?api=1&origin=${userLocation.lat},${userLocation.lng}&destination=${lat},${lng}&travelmode=driving`;
    window.open(googleMapsURL, "_blank"); // Opens Google Maps in a new tab
}

// Fetch Blood Bank Locations Using Overpass API and List Them
function fetchBloodBanks(userLocation, radius) {
    const overpassQuery = `
        [out:json][timeout:25];
        (
            node["amenity"="blood_bank"](around:${radius},${userLocation[0]},${userLocation[1]});
            way["amenity"="blood_bank"](around:${radius},${userLocation[0]},${userLocation[1]});
            relation["amenity"="blood_bank"](around:${radius},${userLocation[0]},${userLocation[1]});
        );
        out center;
    `;
    const overpassURL = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(overpassQuery)}`;

    fetch(overpassURL)
        .then(response => response.json())
        .then(data => {
            clearBloodBankMarkers(); // Remove old markers
            listContainer.innerHTML = "<h3>Nearby Blood Banks</h3>";

            if (data.elements.length === 0 && radius < 100000) {
                console.log(`No results found within ${radius / 1000} km. Expanding search...`);
                fetchBloodBanks(userLocation, radius * 2); // Double the search radius
                return;
            }

            data.elements.forEach(node => {
                let bloodBankLocation = { lat: node.lat, lng: node.lon };
                let bloodBankName = node.tags.name || "Blood Bank";

                let marker = L.marker([bloodBankLocation.lat, bloodBankLocation.lng])
                    .addTo(map)
                    .bindPopup(bloodBankName);

                bloodBankMarkers.push(marker);
                
                let listItem = document.createElement('button');
                listItem.textContent = bloodBankName;
                listItem.style.display = 'block';
                listItem.style.width = '100%';
                listItem.style.margin = '5px 0';
                listItem.style.padding = '10px';
                listItem.style.border = '1px solid #ddd';
                listItem.style.cursor = 'pointer';
                listItem.style.backgroundColor = '#f8f9fa';
                
                // Correctly pass lat & lng to findRoute
                listItem.onclick = () => findRoute(bloodBankLocation.lat, bloodBankLocation.lng);
                
                listContainer.appendChild(listItem);
            });
        })
        .catch(error => {
            console.error("Error fetching blood banks:", error);
            showCentralBloodBanks();
        });
}

// Function to clear previous blood bank markers
function clearBloodBankMarkers() {
    bloodBankMarkers.forEach(marker => map.removeLayer(marker));
    bloodBankMarkers = [];
    listContainer.innerHTML = "";
}



// Show Default Blood Banks in Major Cities if No Nearby Banks Found
function showCentralBloodBanks() {
    let centralBanks = [
        { name: "Red Cross Blood Center", lat: 28.6139, lng: 77.2090 }, // Delhi
        { name: "Mumbai Blood Bank", lat: 19.0760, lng: 72.8777 },
        { name: "Chennai Blood Bank", lat: 13.0827, lng: 80.2707 },
        { name: "Kolkata Blood Center", lat: 22.5726, lng: 88.3639 }
    ];

    centralBanks.forEach(bank => {
        let marker = L.marker([bank.lat, bank.lng])
            .addTo(map)
            .bindPopup(bank.name);

        bloodBankMarkers.push(marker);
    });

    map.setView([20.5937, 78.9629], 5); // Center over India
}


// Find the Nearest Blood Bank
function findNearestBloodBank(userLocation) {
    let nearestBank = null;
    let minDistance = Infinity;

    bloodBankMarkers.forEach(marker => {
        let distance = map.distance(userLocation, marker.getLatLng());
        if (distance < minDistance) {
            minDistance = distance;
            nearestBank = marker.getLatLng();
        }
    });

    return nearestBank;
}

// Find the Fastest Route to the Nearest Blood Bank
function findNearestAndRoute() {
    if (!userMarker) {
        alert("User location not found.");
        return;
    }

    let userLocation = userMarker.getLatLng();
    let nearestBloodBank = findNearestBloodBank(userLocation);

    if (!nearestBloodBank) {
        alert("No blood banks found nearby.");
        return;
    }

    findRoute(nearestBloodBank.lat, nearestBloodBank.lng);
}
