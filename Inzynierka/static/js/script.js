const counters = document.querySelectorAll('.counter'); // Pobranie elementów, których klasa to counter
const speed = 100; // zmienna speed odpowiada za szybkość liczenia
counters.forEach(counter => {
	const updateCount = () => {
		const target = +counter.getAttribute('data-target');
		const count = +counter.innerText;
		const inc = target / speed;
		// Sprawdzenie czy wartość określona w zmiennej target została osiągnięta
		if (count < target) {
			counter.innerText = count + 1000;
			setTimeout(updateCount, 1); //Wywołanie funkcji co 1 milisekundę
		} else {
			counter.innerText = target;
		}
	};
	updateCount();
});