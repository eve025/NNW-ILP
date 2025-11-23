async function sendMessage(){
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if(!text) return;

    // Añadir mensaje del usuario a la ventana
    addMessage(text, 'user');
    input.value = '';

    // Enviar al backend
    const form = new FormData();
    form.append('message', text);
    try{
        const res = await fetch('/predict', { method: 'POST', body: form });
        const j = await res.json();
        if(res.ok){
            // Mostrar respuesta con etiqueta en español y score
            const label = j.prediction_es || j.display || j.prediction || 'Desconocido';
            const score = (typeof j.score === 'number') ? j.score : parseFloat(j.score || 0);
            const threshold = (typeof j.threshold === 'number') ? j.threshold : 0.3;
            // Crear etiqueta coloreada
            const labelSpan = `<span class="result-span ${score>threshold ? 'label-spam' : 'label-legit'}">${label}</span>`;
            const scoreSpan = `<span class="score">(confianza: ${score.toFixed(3)})</span>`;
            const resultado = `${labelSpan} ${scoreSpan}`;
            addMessage(resultado, 'ai');
            // Mostrar consejo si existe
            if (j.advice) {
                addMessage(`<em class="advice">${j.advice}</em>`, 'ai');
            }
        } else {
            addMessage('Error: ' + (j.error || 'Respuesta inesperada'), 'ai');
        }
    } catch(err){
        addMessage('Error de conexión: ' + err.message, 'ai');
    }
}

function addMessage(text, who){
    const win = document.getElementById('chat-window');
    const el = document.createElement('div');
    el.className = (who === 'user') ? 'user-message' : 'ai-message';
    el.innerHTML = text;
    win.appendChild(el);
    win.scrollTop = win.scrollHeight;
}
