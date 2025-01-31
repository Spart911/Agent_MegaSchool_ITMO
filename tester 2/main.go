package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

const (
	serverURL = "http://83.221.210.29:7000/api/request" // замените на URL сервера

)

func postValue(wg *sync.WaitGroup, timeNow string, id int, avg *int32, mu *sync.Mutex) {
	defer wg.Done()
	start := time.Now().Unix()
	data := map[string]interface{}{
		"query": "В каком городе находится главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород",
		"id":    id,
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
		return
	}
	req, err := http.NewRequest("POST", serverURL, bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Ошибка при создании запроса:", err)
		return
	}
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Ошибка при отправке запроса:", err)
		return
	}
	defer resp.Body.Close()
	stop := time.Now().Unix()
	resultTime := stop - start
	atomic.AddInt32(avg, int32(resultTime))
	fmt.Printf("(thread %d ) Ответ сервера: %s\n", id, resp.Status)
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Ошибка при чтении тела ответа:", err)
		return
	}
	bodyString := string(bodyBytes)
	file, err := os.OpenFile(fmt.Sprintf("report_%s.txt", timeNow), os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		fmt.Println("Ошибка при открытии файла:", err)
		return
	}
	defer file.Close()

	// Записываем строку в конец файла
	mu.Lock()
	_, err = file.WriteString(fmt.Sprintf("Ответ сервера:%s\n-----\n%s\n-----\n", resp.Status, bodyString))
	if err != nil {
		fmt.Println("Ошибка при записи в файл:", err)
		return
	}
	mu.Unlock()
	fmt.Printf("(thread %d ) Время выполнения: %d секунд \n", id, resultTime)
}

func main() {
	var wg sync.WaitGroup
	var mu sync.Mutex
	timeNow := time.Now().Format("02012006150405")
	fmt.Print("Введи количество потоков: ")
	var avgTime int32
	var threads int
	_, err := fmt.Scanf("%d", &threads)
	if err != nil {
		fmt.Println("Ошибка при вводе числа потоков:", err)
		return
	}
	numRoutines := threads
	for i := 0; i < numRoutines; i++ {
		wg.Add(1)
		go postValue(&wg, timeNow, i+1, &avgTime, &mu)
	}

	wg.Wait()
	file, err := os.OpenFile(fmt.Sprintf("report_%s.txt", timeNow), os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		fmt.Println("Ошибка при открытии файла:", err)
		return
	}
	defer file.Close()
	_, err = file.WriteString(fmt.Sprintf("\n\n Среднее время выполнения: %d секунд", avgTime/int32(threads)))
	if err != nil {
		fmt.Println("Ошибка при записи в файл:", err)
		return
	}
	_, errExists := os.Stat(fmt.Sprintf("report_%s.txt", timeNow))
	if errExists != nil {
		fmt.Println("Ошибок не обнаружено отчет не сформирован")
		fmt.Println("Нажмите любую клавишу, чтобы выйти...")
		reader := bufio.NewReader(os.Stdin)
		reader.ReadString('\n')
		return
	}
	fmt.Printf("report_%s.txt\n", timeNow)
	fmt.Println("Нажмите любую клавишу, чтобы выйти...")
	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')

}
