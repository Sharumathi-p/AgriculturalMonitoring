console.log("JavaScript file is linked successfully!");

document.addEventListener("DOMContentLoaded", function() {
    function generateRandomValue(min, max, decimals = 2) {
        return (Math.random() * (max - min) + min).toFixed(decimals);
    }

    function updateSensorData() {
        document.getElementById('temperature').textContent = generateRandomValue(18, 35); // Temperature: 18-35°C
        document.getElementById('moisture').textContent = generateRandomValue(25, 70); // Moisture: 25-70%
        document.getElementById('salinity').textContent = generateRandomValue(20, 50); // Salinity: 20-50 ppm
        console.log("✅ Sensor data updated.");
    }

    function checkDiseaseStatus() {
        const probability = Math.random();
        let status;
        
        if (probability > 0.85) {
            status = "⚠️ High risk: Disease detected!";
        } else if (probability > 0.6) {
            status = "⚠️ Moderate risk: Possible disease signs.";
        } else {
            status = "✅ No disease detected.";
        }

        document.getElementById('disease-status').textContent = status;
        console.log("🦠 Disease status checked.");
    }

    function fetchWeatherData() {
        const conditions = ["☀️ Sunny", "🌧 Rainy", "⛅ Partly Cloudy", "🌪 Stormy", "🌤 Clear Sky"];
        const weatherInfo = conditions[Math.floor(Math.random() * conditions.length)] 
            + ` | Temperature: ${generateRandomValue(20, 35)}°C`;
        
        document.getElementById('weather-info').textContent = weatherInfo;
        console.log("🌦 Weather data fetched.");
    }

    document.getElementById('update-sensor').addEventListener("click", updateSensorData);
    document.getElementById('check-disease').addEventListener("click", checkDiseaseStatus);
    document.getElementById('fetch-weather').addEventListener("click", fetchWeatherData);

    setInterval(updateSensorData, 5000);
    setInterval(checkDiseaseStatus, 10000);
    setInterval(fetchWeatherData, 15000);

    updateSensorData();
    checkDiseaseStatus();
    fetchWeatherData();
});
