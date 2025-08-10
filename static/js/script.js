function updateCategoryChart(distribution) {
    var ctx = document.getElementById('categoryChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Car', 'Truck', 'Bus', 'Motorbike'],  // Updated labels
            datasets: [{
                label: 'Number of Vehicles',
                data: [distribution.car, distribution.truck, distribution.bus, distribution.motorbike],  // Updated keys
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
            }]
        }
    });
}
